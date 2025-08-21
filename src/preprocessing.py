from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

from open_instruct.dataset_processor import SFTDatasetProcessor, DatasetConfig


def prepare_dataset(dataset: Dataset | DatasetDict, tokenizer: PreTrainedTokenizer, dataset_config: DatasetConfig, sample_size: int = None) -> Dataset:
    dataset_processor = SFTDatasetProcessor(tokenizer, dataset_config)

    tokenized_dataset = dataset_processor.tokenize(dataset)
    filtered_dataset = dataset_processor.filter(tokenized_dataset)

    subsampled_dataset = filtered_dataset.select(range(sample_size)) if sample_size else filtered_dataset
    subsampled_dataset.set_format(type="pt")

    return subsampled_dataset
