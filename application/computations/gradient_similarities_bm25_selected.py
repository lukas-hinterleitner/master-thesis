from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk
from tqdm.notebook import tqdm
from rank_bm25 import BM25Okapi
from torch.nn import CosineSimilarity

import torch
import random
import gc
import time
import json

import numpy as np

from application.config import hf_model_id, lima_paraphrased_dataset_path, get_dataset_config, get_gradient_similarity_bm25_selected_file_path

from application.utilities.preprocessing import prepare_dataset
from application.utilities.gradient_operations import get_gradients, get_flattened_weight_vector
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
#%%
model.to(device)
model.eval() # set to evaluation because we don't need to update weights
#%%
model.num_parameters()
#%%
dataset = load_from_disk(lima_paraphrased_dataset_path)
#%%
dataset.column_names
#%%
sample_size = 100 # original_dataset.num_rows
sample_size
#%%
original_dataset_config = get_dataset_config(model, sft_messages_key="messages")
original_dataset_config
#%%
paraphrased_dataset_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")
paraphrased_dataset_config
#%%
original_dataset_tokenized = prepare_dataset(dataset=dataset, tokenizer=tokenizer, dataset_config=original_dataset_config, sample_size=sample_size)
#%%
paraphrased_dataset_tokenized = prepare_dataset(dataset=dataset, tokenizer=tokenizer, dataset_config=paraphrased_dataset_config)
#%%
def simple_tokenize(doc: str):
    return doc.split(" ")
#%%
similarity_function = CosineSimilarity(dim=0)
#%%
start_time = time.time()

gradient_similarities = dict()

progress_wrapper = tqdm(original_dataset_tokenized, desc="Calculating gradients and corresponding similarities")

paraphrased_samples = [row["paraphrased_messages"][0]["content"] for row in paraphrased_dataset_tokenized]

for original in progress_wrapper:
    original_id = original["id"]

    original_gradients = get_gradients(model, original, device)
    original_flattened_gradients = get_flattened_weight_vector(original_gradients)
    
    # select most similar samples using bm25
    bm25 = BM25Okapi([simple_tokenize(doc) for doc in paraphrased_samples])
    scores = bm25.get_scores(simple_tokenize(original["messages"][0]["content"]))
    indices = np.argsort((-scores))[:5]
    
    gradient_similarities[original_id] = dict()
    
    for paraphrased in paraphrased_dataset_tokenized.select(indices):

        paraphrased_id = paraphrased["id"]

        progress_wrapper.set_description(desc=f"Processing original ({original_id}) and paraphrased ({paraphrased_id})")        

        paraphrased_gradients = get_gradients(model, paraphrased, device)
        paraphrased_flattened_gradients = get_flattened_weight_vector(paraphrased_gradients)

        similarity = similarity_function(original_flattened_gradients, paraphrased_flattened_gradients).item()
        gradient_similarities[original_id][paraphrased_id] = similarity

progress_wrapper.set_description("Calculating gradients and corresponding similarities")

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

with open(get_gradient_similarity_bm25_selected_file_path(sample_size), "w") as output_file:
    json.dump(gradient_similarities, output_file, indent=4)
