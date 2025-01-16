#%%
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm.notebook import tqdm

import torch
import random
import gc
import time

import numpy as np
import pandas as pd

from utilities.config.model import hf_model_id
from utilities.config.dataset import get_dataset_config
from utilities.config.storage import lima_paraphrased_dataset_path, get_gradient_similarity_file_path

from utilities.preprocessing import prepare_dataset
from utilities.gradient_operations import get_gradients, get_flattened_weight_vector
#%%
# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
#%%
torch.cuda.empty_cache()
gc.collect()
#%%
model = AutoModelForCausalLM.from_pretrained(hf_model_id)
tokenizer = AutoTokenizer.from_pretrained(hf_model_id, return_tensors="pt")

print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)
#%%
use_gpu = True

device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
device

model.to(device)
model.eval() # set to evaluation because we don't need to update weights

model.num_parameters()

dataset = load_from_disk(lima_paraphrased_dataset_path)

dataset.column_names

sample_size = 3 # original_dataset.num_rows
sample_size

original_dataset_config = get_dataset_config(model, sft_messages_key="messages")

paraphrased_dataset_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

original_dataset_tokenized = prepare_dataset(dataset=dataset, tokenizer=tokenizer, dataset_config=original_dataset_config, sample_size=sample_size)

paraphrased_dataset_tokenized = prepare_dataset(dataset=dataset, tokenizer=tokenizer, dataset_config=paraphrased_dataset_config, sample_size=sample_size)

start_time = time.time()

data = []

gradients = dict()

original_ids = set()
paraphrased_ids = set()

progress_wrapper = tqdm(original_dataset_tokenized, desc="Calculating gradients and corresponding similarities")

for original in progress_wrapper:
    original_id = original["id"]
    original_ids.add(original_id)

    original_gradients = get_gradients(model, original, device)
    original_flattened_gradients = get_flattened_weight_vector(original_gradients)

    for paraphrased in paraphrased_dataset_tokenized:
        paraphrased_id = paraphrased["id"]
        paraphrased_ids.add(paraphrased_id)

        progress_wrapper.set_description(desc=f"Processing original ({original_id}) and paraphrased ({paraphrased_id})")

        paraphrased_gradients = get_gradients(model, paraphrased, device)
        paraphrased_flattened_gradients = get_flattened_weight_vector(paraphrased_gradients)

        similarity = original_flattened_gradients.dot(paraphrased_flattened_gradients).item()
        data.append((original_id, paraphrased_id, similarity))

progress_wrapper.set_description("Calculating gradients and corresponding similarities")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

df = pd.DataFrame(data, columns=['original_id', 'paraphrased_id', 'value'])
df_pivot = df.pivot(index='original_id', columns='paraphrased_id', values='value')
df_pivot = df_pivot.reindex(index=sorted(original_ids), columns=sorted(paraphrased_ids))


df_pivot.to_csv(get_gradient_similarity_file_path(sample_size), index=True, header=True)
