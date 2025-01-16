import json
import random

import numpy as np
import torch

from utilities.computation import calculate_bm25_selected_gradient_similarities
from utilities.storage import get_gradient_similarity_bm25_selected_file_path
from utilities.dataset import get_tokenized_datasets
from utilities.model import get_model, get_tokenizer

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = get_model()
tokenizer = get_tokenizer()

original_dataset_tokenized, paraphrased_dataset_tokenized = get_tokenized_datasets(model, tokenizer)

gradient_similarities = calculate_bm25_selected_gradient_similarities(
    original_dataset_tokenized,
    paraphrased_dataset_tokenized,
    model
)

with open(get_gradient_similarity_bm25_selected_file_path(), "w") as output_file:
    json.dump(gradient_similarities, output_file, indent=4)