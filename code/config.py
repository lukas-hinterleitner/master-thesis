import os

from open_instruct.dataset_processor import DatasetConfig
from transformers import PreTrainedModel

def get_dataset_config(model: PreTrainedModel, sft_messages_key="messages", chat_template = "tulu", train_only_on_prompt=True) -> DatasetConfig:
    return DatasetConfig(
        chat_template=chat_template,
        max_token_length=model.config.max_position_embeddings,
        train_only_on_prompt=train_only_on_prompt,
        load_from_cache_file=False,
        batched=True,
        sft_messages_key=sft_messages_key
    )

hf_model_id = "allenai/OLMo-1B-hf"

# "openai-community/gpt2"
# "allenai/OLMo-1B-hf"

data_folder_path = "../data"

lima_dataset_path = os.path.join(data_folder_path, "lima")
lima_paraphrased_dataset_path = os.path.join(data_folder_path, "paraphrased")


gradients_path = os.path.join(data_folder_path, "gradient_similarity")


def get_gradient_similarity_file_path(sample_size):
    return os.path.join(gradients_path, f"sample_size_{sample_size}.csv")


if not os.path.exists(gradients_path):
    os.makedirs(gradients_path)

if not os.path.exists(lima_paraphrased_dataset_path):
    os.makedirs(lima_paraphrased_dataset_path)