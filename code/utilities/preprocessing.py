from datasets import DatasetDict, Dataset
from open_instruct.dataset_processor import CHAT_TEMPLATES, SFTDatasetProcessor, DatasetConfig
from transformers import PreTrainedTokenizerBase

def prepare_dataset(dataset: [Dataset | DatasetDict], tokenizer: PreTrainedTokenizerBase, dataset_config: DatasetConfig, sample_size: int) -> Dataset:
    if not tokenizer.chat_template:
        tokenizer.chat_template = CHAT_TEMPLATES[dataset_config.chat_template]

    dataset_processor = SFTDatasetProcessor(tokenizer, dataset_config)

    sub_sampled_dataset = dataset.select(range(sample_size))

    tokenized_dataset = dataset_processor.tokenize(sub_sampled_dataset)
    filtered_dataset = dataset_processor.filter(tokenized_dataset)

    filtered_dataset.set_format(type="pt")

    return filtered_dataset
