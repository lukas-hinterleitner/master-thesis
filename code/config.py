import os

from open_instruct.dataset_processor import DatasetConfig
from transformers import PreTrainedModel

def get_dataset_config(model: PreTrainedModel, chat_template = "tulu", train_only_on_prompt=True) -> DatasetConfig:
    return DatasetConfig(
        chat_template=chat_template,
        max_token_length=model.config.max_position_embeddings,
        train_only_on_prompt=train_only_on_prompt,
        load_from_cache_file=False
    )

hf_model_id = "allenai/OLMo-1B-hf"

# "openai-community/gpt2"
# "allenai/OLMo-1B-hf"

data_folder_path = "../data"

lima_dataset_path = os.path.join(data_folder_path, "lima")
lima_filtered_dataset_path = os.path.join(lima_dataset_path, "filtered")
lima_filtered_paraphrased_dataset_path = os.path.join(data_folder_path, "paraphrased")

sample_size = 5

gradients_path = os.path.join(data_folder_path, "gradients")

def get_gradients_file_path():
    return os.path.join(gradients_path, f"sample_size_{sample_size}.csv")

if not os.path.exists(gradients_path):
    os.makedirs(gradients_path)

if not os.path.exists(lima_filtered_paraphrased_dataset_path):
    os.makedirs(lima_filtered_paraphrased_dataset_path)