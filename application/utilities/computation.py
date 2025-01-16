import time
import pprint

import datasets
from datasets import Dataset
from torch.nn import CosineSimilarity
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from transformers import PreTrainedModel, PreTrainedTokenizer

from .config.dataset import get_dataset_config
from .gradient_operations import get_gradients, get_flattened_weight_vector
from .preprocessing import prepare_dataset

import numpy as np

datasets.disable_progress_bar()

__amount_comparisons = 5

__USER_TOKEN = "<|user|>\n"
__ASSISTANT_TOKEN = "\n<|assistant|>\n"

def __simple_tokenize(doc: str):
    return doc.split(" ")


def __map_to_message_format(role: str, content: str) -> dict[str, str]:
    return {"role": role, "content": content}


def __generate_model_output_from_paraphrased_row(row: dict, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> dict:
    user_message = list(filter(lambda x: x["role"] == "user", row["paraphrased_messages"]))
    chat_template_applied = tokenizer.apply_chat_template([user_message], return_tensors="pt", add_generation_prompt=True)
    generation = model.generate(chat_template_applied.to(model.device), max_new_tokens=512, do_sample=False)
    decoded = tokenizer.decode(generation[0])

    # Extract assistant message
    end_user = decoded.find(__ASSISTANT_TOKEN)
    start_assistant = end_user + len(__ASSISTANT_TOKEN)
    end_assistant = decoded.find(tokenizer.eos_token)
    assistant_message = decoded[start_assistant:end_assistant].strip()

    row["paraphrased_messages"] = [
        __map_to_message_format("user", user_message[0]["content"]),
        __map_to_message_format("assistant", assistant_message)
    ]

    return row


def calculate_bm25_selected_layer_dot_products(
        original_dataset_tokenized: Dataset,
        paraphrased_dataset_tokenized: Dataset,
        model: PreTrainedModel,
    ) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]], dict[str, dict[str, float]]]:

    original_texts = [row["messages"][0]["content"] for row in original_dataset_tokenized]
    bm25 = BM25Okapi([__simple_tokenize(doc) for doc in original_texts])

    progress_wrapper = tqdm(paraphrased_dataset_tokenized, desc="Calculating layer gradients and dot products between them")

    dot_products = dict()
    paraphrased_dot_products = dict()
    original_dot_products = dict()

    for paraphrased_sample in progress_wrapper:
        paraphrased_id:str = paraphrased_sample["id"]

        paraphrased_gradients = get_gradients(model, paraphrased_sample)

        paraphrased_text = paraphrased_sample["paraphrased_messages"][0]["content"]

        scores = bm25.get_scores(__simple_tokenize(paraphrased_text))
        top_indices = np.argsort(-scores)[:__amount_comparisons]

        # calculate dot products for paraphrased sample
        paraphrased_dot_products[paraphrased_id] = dict()

        for layer, gradient in paraphrased_gradients.items():
            paraphrased_dot_product = (
                gradient
                    .flatten().to(model.device)
                    .dot(gradient
                        .flatten().to(model.device)
                    )
                    .item()
            )

            paraphrased_dot_products[paraphrased_id][layer] = paraphrased_dot_product

        if paraphrased_id in original_dataset_tokenized["id"]:
            matching_original_idx = original_dataset_tokenized["id"].index(paraphrased_id)
            if matching_original_idx not in top_indices:
                top_indices[-1] = matching_original_idx

        dot_products[paraphrased_id] = dict()

        for original_sample in original_dataset_tokenized.select(top_indices):
            original_id:str = original_sample["id"]

            progress_wrapper.set_description(desc=f"Processing paraphrased ({paraphrased_id}) and original ({original_id})")

            original_gradients = get_gradients(model, original_sample)

            # calculate dot products for original sample
            if original_id not in original_dot_products:
                original_dot_products[original_id] = dict()

                for layer, gradient in original_gradients.items():
                    original_dot_product = (
                        gradient
                            .flatten().to(model.device)
                            .dot(gradient
                                .flatten().to(model.device)
                            )
                            .item()
                    )

                    original_dot_products[original_id][layer] = original_dot_product

            # calculate dot products between paraphrased and original sample
            dot_products[paraphrased_id][original_id] = dict()

            for (layer, paraphrased_gradient), (_, original_gradient) in zip(paraphrased_gradients.items(), original_gradients.items()):
                dot_product = (
                    paraphrased_gradient
                        .flatten().to(model.device)
                        .dot(original_gradient
                            .flatten().to(model.device)
                        )
                        .item()
                )

                dot_products[paraphrased_id][original_id][layer] = dot_product

    return dot_products, paraphrased_dot_products, original_dot_products


def calculate_bm25_selected_gradient_similarities(
        original_dataset_tokenized: Dataset,
        paraphrased_dataset_tokenized: Dataset,
        model: PreTrainedModel,
        similarity_function=CosineSimilarity(dim=0)
    ) -> dict[str, dict[str, float]]:
    gradient_similarities = dict()

    original_texts = [row["messages"][0]["content"] for row in original_dataset_tokenized]
    bm25 = BM25Okapi([__simple_tokenize(doc) for doc in original_texts])

    progress_wrapper = tqdm(paraphrased_dataset_tokenized, desc="Calculating gradients and corresponding similarities")

    for paraphrased_sample in progress_wrapper:
        paraphrased_id = paraphrased_sample["id"]

        paraphrased_gradients = get_gradients(model, paraphrased_sample)
        paraphrased_flattened = get_flattened_weight_vector(paraphrased_gradients)

        paraphrased_text = paraphrased_sample["paraphrased_messages"][0]["content"]

        scores = bm25.get_scores(__simple_tokenize(paraphrased_text))
        top_indices = np.argsort(-scores)[:__amount_comparisons]

        if paraphrased_id in original_dataset_tokenized["id"]:
            matching_original_idx = original_dataset_tokenized["id"].index(paraphrased_id)
            if matching_original_idx not in top_indices:
                top_indices[-1] = matching_original_idx

        gradient_similarities[paraphrased_id] = dict()

        for original_sample in original_dataset_tokenized.select(top_indices):
            original_id = original_sample["id"]

            progress_wrapper.set_description(desc=f"Processing paraphrased ({paraphrased_id}) and original ({original_id})")

            # Compute gradient of the original
            original_gradients = get_gradients(model, original_sample)
            original_flattened = get_flattened_weight_vector(original_gradients)

            # Calculate similarity
            similarity = similarity_function(paraphrased_flattened, original_flattened).item()
            gradient_similarities[paraphrased_id][original_id] = similarity

    progress_wrapper.set_description("Finished calculating gradients and similarities")

    return gradient_similarities


def calculate_bm25_selected_model_generated_gradient_similarities(
        original_dataset_tokenized: Dataset,
        paraphrased_dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        similarity_function=CosineSimilarity(dim=0)) -> dict[str, dict[str, float]]:
    gradient_similarities = dict()

    original_texts = [row["messages"][0]["content"] for row in original_dataset_tokenized]
    bm25 = BM25Okapi([__simple_tokenize(doc) for doc in original_texts])

    paraphrased_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

    progress_wrapper = tqdm(paraphrased_dataset, desc="Calculating gradients and corresponding similarities")

    for paraphrased_sample in progress_wrapper:
        paraphrased_id = paraphrased_sample["id"]

        # select most similar original samples using bm25
        scores = bm25.get_scores(__simple_tokenize(paraphrased_sample["paraphrased_messages"][0]["content"]))
        top_indices = np.argsort((-scores))[:__amount_comparisons]

        # check if paraphrased sample is also included
        if paraphrased_id in original_dataset_tokenized["id"]:
            matching_original_idx = original_dataset_tokenized["id"].index(paraphrased_id)
            if matching_original_idx not in top_indices:
                # Replace the last candidate with the ground-truth original
                top_indices[-1] = matching_original_idx

        gradient_similarities[paraphrased_id] = dict()

        model_generated_paraphrased = __generate_model_output_from_paraphrased_row(paraphrased_sample, model, tokenizer)

        # prepare sample for gradient computation
        paraphrased_sample_model_generated = prepare_dataset(Dataset.from_dict({k: [v] for k, v in model_generated_paraphrased.items()}), tokenizer, paraphrased_config)

        paraphrased_gradients = get_gradients(model, paraphrased_sample_model_generated)
        paraphrased_flattened_gradients = get_flattened_weight_vector(paraphrased_gradients)

        for original_sample in original_dataset_tokenized.select(top_indices):
            original_id = original_sample["id"]

            progress_wrapper.set_description(desc=f"Processing paraphrased ({paraphrased_id}) and original ({original_id})")

            original_gradients = get_gradients(model, original_sample)
            original_flattened_gradients = get_flattened_weight_vector(original_gradients)

            similarity = similarity_function(paraphrased_flattened_gradients, original_flattened_gradients).item()
            gradient_similarities[paraphrased_id][original_id] = similarity

    progress_wrapper.set_description("Calculating gradients and corresponding similarities")

    return gradient_similarities