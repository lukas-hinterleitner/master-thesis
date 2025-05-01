from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from .config.device import DEVICE
from .config.model import MODEL_NAME


def get_model(model_name = MODEL_NAME, device = DEVICE) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Device: {device}")

    model.to(device)
    model.eval()  # set to evaluation because we don't need to update weights

    print(f"Model parameters: {model.num_parameters()}")
    print("======================")
    print(model)

    return model

def get_tokenizer(model_name = MODEL_NAME) -> PreTrainedTokenizer:
    return AutoTokenizer.from_pretrained(model_name, return_tensors="pt")

def get_config_model_name() -> str:
    return MODEL_NAME