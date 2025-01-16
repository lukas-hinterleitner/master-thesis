from open_instruct.dataset_processor import DatasetConfig
from transformers import PreTrainedModel

def get_dataset_config(model: PreTrainedModel, sft_messages_key="messages", chat_template = "tulu", train_only_on_prompt=True) -> DatasetConfig:
    return DatasetConfig(
        chat_template=chat_template,
        max_prompt_token_length=model.config.max_position_embeddings,
        max_token_length=model.config.max_position_embeddings,
        train_only_on_prompt=train_only_on_prompt,
        load_from_cache_file=False,
        batched=True,
        sft_messages_key=sft_messages_key
    )