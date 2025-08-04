import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

def get_gradients(model: PreTrainedModel, sample) -> dict[str, torch.Tensor]:
    gradients = {}

    # set gradients to zero so that gradients to not accumulate for each iteration
    model.zero_grad()

    device = model.device

    output = model(
        input_ids=sample["input_ids"].reshape(1, -1).to(device),
        labels=sample["labels"].reshape(1, -1).to(device),
        attention_mask=sample["attention_mask"].reshape(1, -1).to(device),
        use_cache=False
    )

    loss = output.loss

    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is not None:
            gradients[name] = param.grad.clone().detach().cpu()

    return gradients


def get_flattened_weight_vector(weight_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    flattened_weights = []
    for weights in weight_dict.values():
        flattened_weights.append(weights.cpu().flatten())

    return torch.cat(flattened_weights)


__USER_TOKEN = "<|user|>\n"
__ASSISTANT_TOKEN = "\n<|assistant|>\n"


def map_to_message_format(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def generate_model_output_from_paraphrased_sample(sample: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> dict:
    user_message = list(filter(lambda x: x["role"] == "user", sample["paraphrased_messages"]))

    assert len(user_message) == 1, "There should be exactly one user message in the sample."

    chat_template_applied = tokenizer.apply_chat_template([user_message], return_tensors="pt", add_generation_prompt=True)

    generation = model.generate(
        chat_template_applied.to(model.device),
        max_new_tokens=512,
        do_sample=False,
        attention_mask=torch.ones_like(chat_template_applied).to(model.device),
    )

    decoded = tokenizer.decode(generation[0])

    # Extract assistant message
    end_user = decoded.find(__ASSISTANT_TOKEN)
    start_assistant = end_user + len(__ASSISTANT_TOKEN)
    end_assistant = decoded.find(tokenizer.eos_token)
    assistant_message = decoded[start_assistant:end_assistant].strip()

    sample["model_generated_messages"] = [
        map_to_message_format("user", user_message[0]["content"]),
        map_to_message_format("assistant", assistant_message)
    ]

    return sample