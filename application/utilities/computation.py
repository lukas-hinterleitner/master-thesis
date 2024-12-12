import time

from torch.nn import CosineSimilarity
from tqdm import tqdm
from rank_bm25 import BM25Okapi

from .gradient_operations import get_gradients, get_flattened_weight_vector

import numpy as np

def __simple_tokenize(doc: str):
    return doc.split(" ")


def calculate_bm25_selected_gradient_similarities(original_dataset_tokenized, paraphrased_dataset_tokenized, model, similarity_function = CosineSimilarity(dim=0)):
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
        indices = np.argsort((-scores))[:5]
        
        # check if paraphrased sample is also included
        current_sample_idx = paraphrased_dataset_tokenized["id"].index(original_id)
        if current_sample_idx not in indices:
            indices[4] = current_sample_idx # replace least similar of the most similar ones with the actual paraphrase

        gradient_similarities[original_id] = dict()

        for paraphrased in paraphrased_dataset_tokenized.select(indices):
            paraphrased_id = paraphrased["id"]

            progress_wrapper.set_description(desc=f"Processing original ({original_id}) and paraphrased ({paraphrased_id})")

            paraphrased_gradients = get_gradients(model, paraphrased)
            paraphrased_flattened_gradients = get_flattened_weight_vector(paraphrased_gradients)

            similarity = similarity_function(original_flattened_gradients, paraphrased_flattened_gradients).item()
            gradient_similarities[original_id][paraphrased_id] = similarity

    progress_wrapper.set_description("Calculating gradients and corresponding similarities")

    exit(11)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")

    return gradient_similarities