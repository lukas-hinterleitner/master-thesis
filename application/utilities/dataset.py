from transformers import PreTrainedModel, PreTrainedTokenizerBase
from datasets import load_from_disk

from .config.dataset import get_dataset_config
from .config.storage import lima_paraphrased_dataset_path
from .preprocessing import prepare_dataset

def get_tokenized_datasets(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, sample_size: int =5):
    original_dataset_config = get_dataset_config(model, sft_messages_key="messages")
    paraphrased_dataset_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

    dataset = load_from_disk(lima_paraphrased_dataset_path)

    original_dataset_tokenized = prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_config=original_dataset_config,
        sample_size=sample_size
    )

    paraphrased_dataset_tokenized = prepare_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        dataset_config=paraphrased_dataset_config
    )

    return original_dataset_tokenized, paraphrased_dataset_tokenized

