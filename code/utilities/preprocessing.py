from dataclasses import dataclass
from functools import partial

from datasets import DatasetDict, Dataset
from open_instruct.finetune import encode_sft_example
from open_instruct.dataset_processor import CHAT_TEMPLATES
from torch.utils.data import DataLoader, Subset
from transformers import DataCollatorForSeq2Seq, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class LimaDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        ids = [[v for k, v in feature.items() if k == "id"] for feature in features]
        features = [{k: v for k, v in feature.items() if k != "id"} for feature in features]

        batch = super().__call__(features, return_tensors)
        batch["id"] = ids

        return batch

def prepare_dataset(dataset: [Dataset | DatasetDict], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, sample_size: int) -> DataLoader:
    if not tokenizer.chat_template:
        tokenizer.chat_template = CHAT_TEMPLATES["tulu"]

    encode_function = partial(
        encode_sft_example,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )

    sub_sampled_dataset = dataset.select(range(sample_size))

    lm_dataset = sub_sampled_dataset.map(
        encode_function,
        batched=False,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=[
            name
            for name in dataset.column_names
            if name not in ["id", "input_ids", "labels", "attention_mask"]
        ],
        desc="Tokenizing and reformatting instruction data",
    )

    lm_dataset.set_format(type="pt")
    lm_dataset = lm_dataset.filter(lambda example: any(x != -100 for x in example["labels"]))

    train_dataloader = DataLoader(
        lm_dataset,
        shuffle=False,
        collate_fn=LimaDataCollator(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=1,
    )

    return train_dataloader
