from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.config.dataset import get_dataset_config, SAMPLE_SIZE
from src.config.model import MODEL_NAME
from src.preprocessing import prepare_dataset
from src.storage import get_model_generated_huggingface_dataset_path


def get_original_dataset() -> Dataset:
    return load_dataset("lukashinterleitner/LIMA-original", split="train")


def get_paraphrased_dataset(sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    paraphrased = load_dataset("lukashinterleitner/LIMA-paraphrased-GPT-4o-mini", split="train")

    # Handle partition-based subsetting (takes precedence over sample_size)
    if partition_start is not None and partition_end is not None:
        return paraphrased.select(range(partition_start, partition_end))
    elif sample_size:
        return paraphrased.select(range(sample_size))
    else:
        return paraphrased


def get_model_generated_dataset(model_name = MODEL_NAME, sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    try:
        model_generated_dataset = load_dataset(get_model_generated_huggingface_dataset_path(model_name), split="train")
    except Exception:
        raise RuntimeError(f"Error loading model-generated dataset for model: {model_name}. Please create the model-generated dataset first using --dataset-type model-generated")

    # Handle partition-based subsetting (takes precedence over sample_size)
    if partition_start is not None and partition_end is not None:
        return model_generated_dataset.select(range(partition_start, partition_end))
    elif sample_size:
        return model_generated_dataset.select(range(sample_size))
    else:
        return model_generated_dataset


def get_paraphrased_dataset_tokenized(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    paraphrased_dataset = get_paraphrased_dataset(sample_size, partition_start, partition_end)
    paraphrased_dataset_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

    return prepare_dataset(
        dataset=paraphrased_dataset,
        tokenizer=tokenizer,
        dataset_config=paraphrased_dataset_config,
    )


def get_original_dataset_tokenized(model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dataset:
    original_dataset_config = get_dataset_config(model, sft_messages_key="messages")
    dataset = get_original_dataset() # get full original dataset

    return prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_config=original_dataset_config,
    )


def get_model_generated_dataset_tokenized(model: PreTrainedModel, tokenizer: PreTrainedTokenizer, sample_size = SAMPLE_SIZE, partition_start=None, partition_end=None) -> Dataset:
    model_generated_dataset = get_model_generated_dataset(model_name=MODEL_NAME, sample_size=sample_size, partition_start=partition_start, partition_end=partition_end)
    model_generated_dataset_config = get_dataset_config(model, sft_messages_key="model_generated_messages")

    return prepare_dataset(
        dataset=model_generated_dataset,
        tokenizer=tokenizer,
        dataset_config=model_generated_dataset_config,
    )


def get_samples(sample_ids: list[str]) -> Dataset:
    original_dataset = get_original_dataset()
    paraphrased_dataset = get_paraphrased_dataset(sample_size=None, partition_start=None, partition_end=None)
    model_generated_dataset = get_model_generated_dataset(sample_size=None, partition_start=None, partition_end=None)

    assert original_dataset["id"] == paraphrased_dataset["id"] == model_generated_dataset["id"], "IDs don't align across datasets"

    merged = concatenate_datasets(
        [
            original_dataset,
            paraphrased_dataset.remove_columns("id"),
            model_generated_dataset.remove_columns("id"),
        ],
        axis=1
    )

    return merged.filter(lambda example: example["id"] in sample_ids)

