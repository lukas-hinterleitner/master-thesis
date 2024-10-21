from functools import partial

from datasets import DatasetDict, Dataset
from open_instruct.finetune import encode_sft_example
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, PreTrainedModel, PreTrainedTokenizerBase


def prepare_dataset(dataset: [Dataset | DatasetDict], model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> DataLoader:
    encode_function = partial(
        encode_sft_example,
        tokenizer=tokenizer,
        max_seq_length=2048,
    )

    lm_dataset = dataset.map(
        encode_function,
        batched=False,
        num_proc=1,
        load_from_cache_file=False,
        remove_columns=[
            name
            for name in dataset.column_names
            if name not in ["input_ids", "labels", "attention_mask"]
        ],
        desc="Tokenizing and reformatting instruction data",
    )
    lm_dataset.set_format(type="pt")
    lm_dataset = lm_dataset.filter(lambda example: (example["labels"] != -100).any())

    train_dataloader = DataLoader(
        lm_dataset,
        shuffle=False,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
        batch_size=1,
    )

    return train_dataloader
