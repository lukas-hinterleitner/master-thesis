import random

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config.model import MODEL_NAME
from src.config.dataset import get_dataset_config
from src.config.storage import lima_paraphrased_dataset_path

from src.model_operations import get_gradients
from src.preprocessing import prepare_dataset

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(torch.cuda.get_device_name(0))

model.to(device)
model.eval()

dataset_config = get_dataset_config(model)

dataset = load_from_disk(lima_paraphrased_dataset_path)
train_dataloader = prepare_dataset(dataset=dataset, dataset_config=dataset_config, tokenizer=tokenizer, sample_size=2)

training_sample_0 = list(train_dataloader)[0]
training_sample_1 = list(train_dataloader)[1]

# first and foremost, check if two samples are different after tokenizing
if training_sample_0["input_ids"].equal(training_sample_1["input_ids"]):
    print("Tokenized inputs are the same. Check tokenizing functionality!")
else:
    print("As expected, tokenized inputs are not the same. ")

# check idempotency of some input samples
gradients_sample_0 = get_gradients(model, training_sample_0, device)
gradients_sample_1 = get_gradients(model, training_sample_1, device)
gradients_sample_0_later = get_gradients(model, training_sample_0, device)

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