import time

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


def __generate_model_output_from_paraphrased_inputs(dataset: Dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Dataset:
    def mapping_function(row):
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

    return dataset.map(
        mapping_function,
    )


def calculate_bm25_selected_gradient_similarities(
        original_dataset_tokenized: Dataset,
        paraphrased_dataset_tokenized: Dataset,
        model: PreTrainedModel,
        similarity_function = CosineSimilarity(dim=0)) -> dict[str, dict[str, float]]:
    start_time = time.time()

    gradient_similarities = dict()

    progress_wrapper = tqdm(original_dataset_tokenized, desc="Calculating gradients and corresponding similarities")

    paraphrased_samples = [row["paraphrased_messages"][0]["content"] for row in paraphrased_dataset_tokenized]

    for original in progress_wrapper:
        original_id = original["id"]

        original_gradients = get_gradients(model, original)
        original_flattened_gradients = get_flattened_weight_vector(original_gradients)

        # select most similar samples using bm25
        bm25 = BM25Okapi([__simple_tokenize(doc) for doc in paraphrased_samples])
        scores = bm25.get_scores(__simple_tokenize(original["messages"][0]["content"]))
        indices = np.argsort((-scores))[:__amount_comparisons]
        
        # check if paraphrased sample is also included
        current_sample_idx = paraphrased_dataset_tokenized["id"].index(original_id)
        if current_sample_idx not in indices:
            indices[__amount_comparisons-1] = current_sample_idx # replace least similar of the most similar ones with the actual paraphrase

        gradient_similarities[original_id] = dict()

        for paraphrased in paraphrased_dataset_tokenized.select(indices):
            paraphrased_id = paraphrased["id"]

            progress_wrapper.set_description(desc=f"Processing original ({original_id}) and paraphrased ({paraphrased_id})")

            paraphrased_gradients = get_gradients(model, paraphrased)
            paraphrased_flattened_gradients = get_flattened_weight_vector(paraphrased_gradients)

            similarity = similarity_function(original_flattened_gradients, paraphrased_flattened_gradients).item()
            gradient_similarities[original_id][paraphrased_id] = similarity

    progress_wrapper.set_description("Calculating gradients and corresponding similarities")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return gradient_similarities


def calculate_bm25_selected_model_generated_gradient_similarities(
        original_dataset_tokenized: Dataset,
        paraphrased_dataset: Dataset,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        similarity_function=CosineSimilarity(dim=0)) -> dict[str, dict[str, float]]:
    start_time = time.time()

    gradient_similarities = dict()

    progress_wrapper = tqdm(original_dataset_tokenized, desc="Calculating gradients and corresponding similarities")

    paraphrased_samples = [row["paraphrased_messages"][0]["content"] for row in paraphrased_dataset]
    paraphrased_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

    for original in progress_wrapper:
        original_id = original["id"]

        original_gradients = get_gradients(model, original)
        original_flattened_gradients = get_flattened_weight_vector(original_gradients)

        # select most similar samples using bm25
        bm25 = BM25Okapi([__simple_tokenize(doc) for doc in paraphrased_samples])
        scores = bm25.get_scores(__simple_tokenize(original["messages"][0]["content"]))
        indices = np.argsort((-scores))[:__amount_comparisons]

        # check if paraphrased sample is also included
        current_sample_idx = paraphrased_dataset["id"].index(original_id)
        if current_sample_idx not in indices:
            indices[
                __amount_comparisons - 1] = current_sample_idx  # replace least similar of the most similar ones with the actual paraphrase

        gradient_similarities[original_id] = dict()

        model_generated_paraphrases_dataset = __generate_model_output_from_paraphrased_inputs(paraphrased_dataset.select(indices), model, tokenizer)

        for paraphrased in prepare_dataset(model_generated_paraphrases_dataset, tokenizer, paraphrased_config):
            paraphrased_id = paraphrased["id"]

            progress_wrapper.set_description(desc=f"Processing original ({original_id}) and paraphrased ({paraphrased_id})")

            paraphrased_gradients = get_gradients(model, paraphrased)
            paraphrased_flattened_gradients = get_flattened_weight_vector(paraphrased_gradients)

            similarity = similarity_function(original_flattened_gradients, paraphrased_flattened_gradients).item()
            gradient_similarities[original_id][paraphrased_id] = similarity

    progress_wrapper.set_description("Calculating gradients and corresponding similarities")

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return gradient_similarities