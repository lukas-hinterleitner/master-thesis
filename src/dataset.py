from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import load_from_disk, Dataset

from src.config.dataset import get_dataset_config, SAMPLE_SIZE
from src.config.storage import lima_paraphrased_dataset_path
from src.preprocessing import prepare_dataset

def get_tokenized_datasets(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, partition_start=None, partition_end=None) -> tuple[Dataset, Dataset]:
    original_dataset_tokenized = get_original_dataset_tokenized(model, tokenizer)

    paraphrased_dataset_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

    # Determine which subset of data to use
    if partition_start is not None and partition_end is not None:
        # Use the specified partition
        subset = original_dataset_tokenized.select_columns(["id", "paraphrased_messages"]).select(range(partition_start, partition_end))
    elif SAMPLE_SIZE:
        # Use sample size if partitioning not specified
        subset = original_dataset_tokenized.select_columns(["id", "paraphrased_messages"]).select(range(SAMPLE_SIZE))
    else:
        # Use all data
        subset = original_dataset_tokenized.select_columns(["id", "paraphrased_messages"])

    paraphrased_dataset_tokenized = prepare_dataset(
        dataset=subset,
        tokenizer=tokenizer,
        dataset_config=paraphrased_dataset_config
    )

    return original_dataset_tokenized.remove_columns(["paraphrased_messages"]), paraphrased_dataset_tokenized


def get_original_dataset_tokenized(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    original_dataset_config = get_dataset_config(model, sft_messages_key="messages")
    dataset = load_from_disk(lima_paraphrased_dataset_path)

    return prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_config=original_dataset_config,
    )

def get_paraphrased_dataset(sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    paraphrased = load_from_disk(lima_paraphrased_dataset_path).select_columns(["id", "paraphrased_messages"])

    # Handle partition-based subsetting (takes precedence over sample_size)
    if partition_start is not None and partition_end is not None:
        return paraphrased.select(range(partition_start, partition_end))
    elif sample_size:
        return paraphrased.select(range(sample_size))
    else:
        return paraphrased

def get_samples(sample_ids: list[str]) -> Dataset:
    return load_from_disk(lima_paraphrased_dataset_path).filter(lambda example: example["id"] in sample_ids)

