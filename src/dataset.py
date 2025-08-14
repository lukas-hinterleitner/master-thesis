from datasets import Dataset, load_dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.config.dataset import get_dataset_config, SAMPLE_SIZE
from src.config.model import MODEL_NAME
from src.config.storage import lima_paraphrased_dataset_path
from src.preprocessing import prepare_dataset
from src.storage import get_model_generated_huggingface_dataset_path


def get_paraphrased_dataset_tokenized(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    paraphrased_dataset = get_paraphrased_dataset(sample_size, partition_start, partition_end)

    paraphrased_dataset_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

    return prepare_dataset(
        dataset=paraphrased_dataset,
        tokenizer=tokenizer,
        dataset_config=paraphrased_dataset_config,
    )


def get_original_dataset_tokenized(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> Dataset:
    original_dataset_config = get_dataset_config(model, sft_messages_key="messages")
    dataset = get_original_dataset(None, None, None) # get full original dataset

    return prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_config=original_dataset_config,
    )


def get_model_generated_dataset_tokenized(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    model_generated_dataset = get_model_generated_dataset(model_name=MODEL_NAME, sample_size=sample_size, partition_start=partition_start, partition_end=partition_end)

    model_generated_dataset_config = get_dataset_config(model, sft_messages_key="model_generated_messages")

    return prepare_dataset(
        dataset=model_generated_dataset,
        tokenizer=tokenizer,
        dataset_config=model_generated_dataset_config,
    )

def get_original_dataset(sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    original_dataset = load_dataset("lukashinterleitner/LIMA-paraphrased-GPT-4o-mini", split="train").select_columns(["id", "messages"])

    # Handle partition-based subsetting (takes precedence over sample_size)
    if partition_start is not None and partition_end is not None:
        return original_dataset.select(range(partition_start, partition_end))
    elif sample_size:
        return original_dataset.select(range(sample_size))
    else:
        return original_dataset

def get_paraphrased_dataset(sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    paraphrased = load_dataset("lukashinterleitner/LIMA-paraphrased-GPT-4o-mini", split="train").select_columns(["id", "paraphrased_messages"])

    # Handle partition-based subsetting (takes precedence over sample_size)
    if partition_start is not None and partition_end is not None:
        return paraphrased.select(range(partition_start, partition_end))
    elif sample_size:
        return paraphrased.select(range(sample_size))
    else:
        return paraphrased

def get_model_generated_dataset(model_name = MODEL_NAME, sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    try:
        path = get_model_generated_huggingface_dataset_path(model_name)
        model_generated_dataset = load_dataset(path, split="train").select_columns(["id", "model_generated_messages"])
    except Exception as e:
        # raise error
        raise RuntimeError(f"Error loading model-generated dataset from {path}. Please create the model-generated dataset first using --dataset-type model-generated")

    # Handle partition-based subsetting (takes precedence over sample_size)
    if partition_start is not None and partition_end is not None:
        return model_generated_dataset.select(range(partition_start, partition_end))
    elif sample_size:
        return model_generated_dataset.select(range(sample_size))
    else:
        return model_generated_dataset

def get_samples(sample_ids: list[str]) -> Dataset:
    return get_model_generated_dataset(None, None, None).filter(lambda example: example["id"] in sample_ids)

