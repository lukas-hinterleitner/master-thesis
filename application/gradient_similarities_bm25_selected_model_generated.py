import json
import random

import numpy as np
import torch

from utilities.computation import calculate_bm25_selected_model_generated_gradient_similarities
from utilities.config.storage import get_gradient_similarity_bm25_selected_model_generated_file_path
from utilities.dataset import get_original_dataset_tokenized, get_paraphrased_dataset
from utilities.model import get_model, get_tokenizer

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = get_model(use_gpu=True)
tokenizer = get_tokenizer()

sample_size = 100

original_dataset_tokenized = get_original_dataset_tokenized(model, tokenizer)
paraphrased_dataset = get_paraphrased_dataset(sample_size)

gradient_similarities = calculate_bm25_selected_model_generated_gradient_similarities(
    original_dataset_tokenized,
    paraphrased_dataset,
    model,
    tokenizer
)

with open(get_gradient_similarity_bm25_selected_model_generated_file_path(sample_size), "w") as output_file:
    json.dump(gradient_similarities, output_file, indent=4)