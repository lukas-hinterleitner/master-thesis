import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .config.model import hf_model_id

def get_model(use_gpu=True) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(hf_model_id)

    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    print(f"Device: {device}")

    model.to(device)
    model.eval()  # set to evaluation because we don't need to update weights

    print(f"Model parameters: {model.num_parameters()}")
    print("======================")
    print(model)

    return model

def get_tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(hf_model_id, return_tensors="pt")