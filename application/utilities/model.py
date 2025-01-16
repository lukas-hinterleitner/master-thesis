from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .config.device import DEVICE
from .config.model import MODEL_NAME


def get_model() -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    print(f"Device: {DEVICE}")

    model.to(DEVICE)
    model.eval()  # set to evaluation because we don't need to update weights

    print(f"Model parameters: {model.num_parameters()}")
    print("======================")
    print(model)

    return model

def get_tokenizer() -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(MODEL_NAME, return_tensors="pt")

def get_model_name() -> str:
    return MODEL_NAME