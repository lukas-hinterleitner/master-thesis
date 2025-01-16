import json
import random

import numpy as np
import torch

from utilities.computation import calculate_bm25_selected_layer_dot_products
from utilities.storage import get_dot_product_bm25_selected_file_path
from utilities.dataset import get_tokenized_datasets
from utilities.model import get_model, get_tokenizer

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = get_model(use_gpu=True)
tokenizer = get_tokenizer()

sample_size = 3

original_dataset_tokenized, paraphrased_dataset_tokenized = get_tokenized_datasets(model, tokenizer, sample_size)

dot_products, paraphrased_dot_products, original_dot_products = calculate_bm25_selected_layer_dot_products(original_dataset_tokenized, paraphrased_dataset_tokenized, model)

# store dot products
with open(get_dot_product_bm25_selected_file_path(model.name_or_path, sample_size, "dot_products"), "w") as output_file:
    json.dump(dot_products, output_file, indent=4)

with open(get_dot_product_bm25_selected_file_path(model.name_or_path, sample_size, "paraphrased_dot_products"), "w") as output_file:
    json.dump(paraphrased_dot_products, output_file, indent=4)

with open(get_dot_product_bm25_selected_file_path(model.name_or_path, sample_size, "original_dot_products"), "w") as output_file:
    json.dump(original_dot_products, output_file, indent=4)