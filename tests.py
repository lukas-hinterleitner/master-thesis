import random

import numpy as np
import torch

from src.dataset import get_original_dataset_tokenized
from src.model import get_model, get_tokenizer
from src.model_operations import get_gradients

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = get_model()
tokenizer = get_tokenizer()

dataset = get_original_dataset_tokenized(model, tokenizer)

training_sample_0 = dataset[0]
training_sample_1 = dataset[1]

# first and foremost, check if two samples are different after tokenizing
if training_sample_0["input_ids"].equal(training_sample_1["input_ids"]):
    print("Tokenized inputs are the same. Check tokenizing functionality!")
else:
    print("As expected, tokenized inputs are not the same. ")

# check idempotency of some input samples
gradients_sample_0 = get_gradients(model, training_sample_0)
gradients_sample_1 = get_gradients(model, training_sample_1)
gradients_sample_0_later = get_gradients(model, training_sample_0)

# gradient dictionary keys of sample_0, sample_0_later and sample_1 should be the same
assert gradients_sample_0.keys() == gradients_sample_0_later.keys() == gradients_sample_1.keys(), "Gradient dictionaries must have same keys."

# compare gradients of the same sample
for key in gradients_sample_0.keys():
    assert gradients_sample_0[key].equal(gradients_sample_0_later[key]), f"Gradient '{key}' not equal!"

print("Gradients are equal when using the same sample.")

# compare gradients of two different samples
for key in gradients_sample_0.keys():
    assert not gradients_sample_0[key].equal(gradients_sample_1[key]), f"Gradient '{key}' equal!"

print("Gradients are different when using two different samples.")
print("All tests passed.")