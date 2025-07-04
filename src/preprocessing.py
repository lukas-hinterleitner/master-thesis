from datasets import DatasetDict, Dataset
from transformers import PreTrainedTokenizerBase

from open_instruct.dataset_processor import CHAT_TEMPLATES, SFTDatasetProcessor, DatasetConfig

def get_chat_template(dataset_config: DatasetConfig) -> str:
    if not dataset_config.chat_template:
        return CHAT_TEMPLATES["tulu"] # Default chat template if not specified

    return CHAT_TEMPLATES[dataset_config.chat_template]

def prepare_dataset(dataset: Dataset | DatasetDict, tokenizer: PreTrainedTokenizerBase, dataset_config: DatasetConfig, sample_size: int = None) -> Dataset:
    if not tokenizer.chat_template:
        tokenizer.chat_template = get_chat_template(dataset_config)

    dataset_processor = SFTDatasetProcessor(tokenizer, dataset_config)

    tokenized_dataset = dataset_processor.tokenize(
        dataset.select(range(sample_size)) if sample_size
        else dataset
    )

    tokenized_dataset.set_format(type="pt")

    #filtered_dataset = dataset_processor.filter(tokenized_dataset)
    #filtered_dataset.set_format(type="pt")

    return tokenized_dataset
