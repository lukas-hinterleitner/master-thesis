from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import load_from_disk, Dataset

from .config.dataset import get_dataset_config, SAMPLE_SIZE
from .config.storage import lima_paraphrased_dataset_path
from .preprocessing import prepare_dataset

def get_tokenized_datasets(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> tuple[Dataset, Dataset]:
    original_dataset_tokenized = get_original_dataset_tokenized(model, tokenizer)

    paraphrased_dataset_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

    paraphrased_dataset_tokenized = prepare_dataset(
        dataset=original_dataset_tokenized.select_columns(["id", "paraphrased_messages"]),
        tokenizer=tokenizer,
        dataset_config=paraphrased_dataset_config,
        sample_size=SAMPLE_SIZE
    )

    return original_dataset_tokenized.remove_columns(["paraphrased_messages"]), paraphrased_dataset_tokenized


def get_original_dataset_tokenized(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    original_dataset_config = get_dataset_config(model, sft_messages_key="messages")
    dataset = load_from_disk(lima_paraphrased_dataset_path)

    return prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_config=original_dataset_config,
        sample_size=SAMPLE_SIZE
    )

def get_paraphrased_dataset():
    paraphrased = load_from_disk(lima_paraphrased_dataset_path).select_columns(["id", "paraphrased_messages"])

    if SAMPLE_SIZE:
        return paraphrased.select(range(SAMPLE_SIZE))
    else:
        return paraphrased
