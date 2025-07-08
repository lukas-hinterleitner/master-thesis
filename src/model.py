from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.config.device import DEVICE
from src.config.model import MODEL_NAME
from src.config.dataset import get_chat_template


def get_model(model_name = MODEL_NAME, device = DEVICE) -> PreTrainedModel:
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print(f"Device: {device}")

    model.to(device)
    model.eval()  # set to evaluation because we don't need to update weights

    print(f"Model parameters: {model.num_parameters()}")
    print("=" * 50)
    print(model)

    return model

def get_tokenizer(model_name = MODEL_NAME) -> PreTrainedTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name, return_tensors="pt")
    tokenizer.chat_template = get_chat_template()

    return tokenizer

def get_config_model_name() -> str:
    return MODEL_NAME

def get_num_parameters_per_layer(model: PreTrainedModel) -> dict[str, int]:
    num_parameters_per_layer = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_parameters_per_layer[name] = param.numel()

    return num_parameters_per_layer