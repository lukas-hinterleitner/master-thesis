import dotenv
import os

from open_instruct.dataset_processor import DatasetConfig, CHAT_TEMPLATES
from transformers import PreTrainedModel

if not dotenv.load_dotenv():
    raise Exception("Couldn't load .env file. Please make sure that the .env file exists according to the README.md file.")

SAMPLE_SIZE = int(os.getenv("MT_SAMPLE_SIZE")) if os.getenv("MT_SAMPLE_SIZE") else None

DEFAULT_CHAT_TEMPLATE = "tulu"

def get_dataset_config(model: PreTrainedModel, sft_messages_key="messages", chat_template = DEFAULT_CHAT_TEMPLATE, train_only_on_prompt=True) -> DatasetConfig:
    return DatasetConfig(
        chat_template=chat_template,
        max_token_length=model.config.max_position_embeddings,
        train_only_on_prompt=train_only_on_prompt,
        load_from_cache_file=True,
        batched=True,
        sft_messages_key=sft_messages_key
    )

def get_chat_template(dataset_config: DatasetConfig = None) -> str:
    if dataset_config and dataset_config.chat_template:
        return CHAT_TEMPLATES[dataset_config.chat_template]

    return CHAT_TEMPLATES[DEFAULT_CHAT_TEMPLATE] # Default chat template if not specified

