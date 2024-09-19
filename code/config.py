import os

hf_model_id = "allenai/OLMo-1B-hf"

data_folder_path = "../data"

lima_dataset_path = os.path.join(data_folder_path, "lima")
lima_filtered_dataset_path = os.path.join(lima_dataset_path, "filtered")
lima_filtered_paraphrased_dataset_path = os.path.join(data_folder_path, "paraphrased")

if not os.path.exists(lima_filtered_paraphrased_dataset_path):
    os.makedirs(lima_filtered_paraphrased_dataset_path)