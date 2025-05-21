import datasets
import numpy as np
from tqdm import tqdm
import torch
from datasets import Dataset
from rank_bm25 import BM25Okapi
from torch.nn import CosineSimilarity
from transformers import PreTrainedModel, PreTrainedTokenizer

from trak.projectors import CudaProjector, ProjectionType

from .config.dataset import get_dataset_config
from .model_operations import get_gradients, get_flattened_weight_vector, generate_model_output_from_paraphrased_sample
from .preprocessing import prepare_dataset

datasets.disable_progress_bar()

__amount_comparisons = 5

def __simple_tokenize(doc: str):
    return doc.split(" ")


def build_bm25_index(original_dataset_tokenized: Dataset) -> BM25Okapi:
    """
    Build and return a BM25 index using the 'content' of the original dataset.
    """
    original_texts = [row["messages"][0]["content"] for row in original_dataset_tokenized]
    return BM25Okapi([__simple_tokenize(doc) for doc in original_texts])


def select_top_bm25_matches(
    query_text: str,
    bm25: BM25Okapi,
    original_dataset_tokenized: Dataset,
    paraphrased_id: str,
    top_k: int
) -> list[int]:
    """
    Given a query_text, returns the indices in original_dataset_tokenized
    that match best under BM25, ensuring that if paraphrased_id is present
    in the original dataset, that exact original is included in the top set.
    """
    scores = bm25.get_scores(__simple_tokenize(query_text))
    top_indices = np.argsort(-scores)[:top_k]  # top_k matches

    # If there's a "ground truth" original with the same ID, ensure it is included
    if paraphrased_id in original_dataset_tokenized["id"]:
        matching_original_idx = original_dataset_tokenized["id"].index(paraphrased_id)
        if matching_original_idx not in top_indices:
            top_indices[-1] = matching_original_idx

    return top_indices.tolist()


def get_paraphrased_sample_gradients(
    paraphrased_sample: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer = None,
    is_model_generated: bool = False,
    paraphrased_config = None
) -> dict[str, torch.Tensor]:
    """
    Depending on `is_model_generated`, either:
      - use the sample's existing paraphrased text
      - or generate a new model output from the paraphrased row,
        then prepare a dataset row for gradient computation
    """
    if is_model_generated:
        # 1. Generate text from the model
        model_generated_paraphrased = generate_model_output_from_paraphrased_sample(
            paraphrased_sample, model, tokenizer
        )
        # 2. Convert it into a tokenized dataset sample
        paraphrased_sample_model_generated = prepare_dataset(
            Dataset.from_dict({k: [v] for k, v in model_generated_paraphrased.items()}),
            tokenizer,
            paraphrased_config
        )
        # 3. Compute gradients
        gradients = get_gradients(model, paraphrased_sample_model_generated)
    else:
        # Use the paraphrased_dataset_tokenized's sample as-is
        gradients = get_gradients(model, paraphrased_sample)

    return gradients


def compute_flattened_similarity(
    gradients1: dict[str, torch.Tensor],
    gradients2: dict[str, torch.Tensor],
    similarity_function = CosineSimilarity(dim=0),
    device: torch.device = torch.device("cpu")
) -> float:
    """
    Flatten and compute a single similarity score (cosine or otherwise)
    between the entire sets of gradients.
    """
    grad1_flat = get_flattened_weight_vector(gradients1).to(device)
    grad2_flat = get_flattened_weight_vector(gradients2).to(device)
    return similarity_function(grad1_flat, grad2_flat).item()


def compute_layerwise_dot_products(
    gradients1: dict[str, torch.Tensor],
    gradients2: dict[str, torch.Tensor],
    device: torch.device = torch.device("cpu")
) -> dict[str, float]:
    """
    Compute layer-by-layer dot product (grad1.layer_i dot grad2.layer_i).
    Returns dict: layer -> dot product.
    """
    out = {}
    for (layer, grad1), (_, grad2) in zip(gradients1.items(), gradients2.items(), strict=True):
        out[layer] = grad1.flatten().to(device).dot(grad2.flatten().to(device)).item()
    return out

def compute_layerwise_self_dot_products(
    gradients: dict[str, torch.Tensor],
    device: torch.device = torch.device("cpu")
) -> dict[str, float]:
    """
    For each layer, compute grad(layer) dot grad(layer).
    """
    out = {}
    for layer, grad in gradients.items():
        flat = grad.flatten().to(device)
        out[layer] = flat.dot(flat).item()
    return out


def calculate_paraphrased_gradient_similarities(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset_tokenized: Dataset,
    model: PreTrainedModel,
    top_k: int = __amount_comparisons,
) -> dict[str, dict[str, float]]:
    bm25 = build_bm25_index(original_dataset_tokenized)
    gradient_similarities: dict[str, dict[str, float]] = {}

    progress = tqdm(paraphrased_dataset_tokenized, desc="Calculating gradients + similarities")

    for paraphrased_sample in progress:
        paraphrased_id = paraphrased_sample["id"]
        # 1) Compute paraphrased gradients
        paraphrased_grad = get_paraphrased_sample_gradients(
            paraphrased_sample,
            model,
            tokenizer=None,
            is_model_generated=False
        )
        # 2) BM25 top matches
        paraphrased_text = paraphrased_sample["paraphrased_messages"][0]["content"]
        top_indices = select_top_bm25_matches(
            paraphrased_text,
            bm25,
            original_dataset_tokenized,
            paraphrased_id,
            top_k
        )
        # 3) Loop over top matches and compute similarity
        gradient_similarities[paraphrased_id] = {}
        for original_sample in original_dataset_tokenized.select(top_indices):
            original_id = original_sample["id"]

            progress.set_description(f"P({paraphrased_id}) vs O({original_id})")

            original_grad = get_gradients(model, original_sample)
            sim = compute_flattened_similarity(
                paraphrased_grad, original_grad
            )
            gradient_similarities[paraphrased_id][original_id] = sim

    progress.set_description("Finished calculating flattened similarities")
    return gradient_similarities


def calculate_paraphrased_random_projected_gradient_similarities(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset_tokenized: Dataset,
    model: PreTrainedModel,
    top_k: int = __amount_comparisons,
) -> dict[str, dict[str, dict[int, float]]]:
    bm25 = build_bm25_index(original_dataset_tokenized)
    gradient_similarities_random_projected: dict[str, dict[str, dict[int, float]]] = {}

    progress = tqdm(paraphrased_dataset_tokenized, desc="Calculating gradients + similarities using random projection")
    similarity_function = CosineSimilarity(dim=0)

    projection_dimensions = [
        int(model.num_parameters() * 0.01),
        int(model.num_parameters() * 0.05),
    ]

    for projection_dim in projection_dimensions:
        print(f"Projection dimension: {projection_dim}")

        for paraphrased_sample in progress:
            paraphrased_id = paraphrased_sample["id"]
            # 1) Compute paraphrased gradients
            paraphrased_grads = get_paraphrased_sample_gradients(
                paraphrased_sample,
                model,
                tokenizer=None,
                is_model_generated=False
            )

            # 2) BM25 top matches
            paraphrased_text = paraphrased_sample["paraphrased_messages"][0]["content"]
            top_indices = select_top_bm25_matches(
                paraphrased_text,
                bm25,
                original_dataset_tokenized,
                paraphrased_id,
                top_k
            )

            paraphrased_grads_flattened = get_flattened_weight_vector(paraphrased_grads)

            projector = CudaProjector(
                grad_dim=paraphrased_grads_flattened.numel(),
                proj_dim=projection_dim,
                seed=42,
                device=model.device,
                proj_type=ProjectionType.rademacher,
                max_batch_size=8
            )

            print(f"Paraphrased gradient shape: {paraphrased_grads_flattened.shape}")
            print(f"Paraphrased gradient reshaped shape: {paraphrased_grads_flattened.reshape(1, -1).shape}")

            down_projected_paraphrased_gradient = projector.project(
                grads=paraphrased_grads_flattened.half().cuda(model.device),
                model_id=1
            ).cpu()

            print(f"Down projected paraphrased gradient shape: {down_projected_paraphrased_gradient.shape}")

            # 3) Loop over top matches and compute similarity
            gradient_similarities_random_projected[paraphrased_id] = {}
            for original_sample in original_dataset_tokenized.select(top_indices):
                original_id = original_sample["id"]

                progress.set_description(f"P({paraphrased_id}) vs O({original_id})")

                original_grads = get_gradients(model, original_sample)

                gradient_similarities_random_projected[paraphrased_id][original_id] = {}

                # get flattened vector of the original gradients
                original_grads_flattened = get_flattened_weight_vector(original_grads)

                print(f"Original gradient shape: {original_grads_flattened.shape}")
                print(f"Original gradient reshaped shape: {original_grads_flattened.reshape(1, -1).shape}")

                down_projected_original_gradient = projector.project(
                    grads=original_grads_flattened.half().cuda(model.device),
                    model_id=0
                ).cpu()

                print(f"Down projected original gradient shape: {down_projected_original_gradient.shape}")

                similarity = similarity_function(
                    down_projected_paraphrased_gradient.flatten().cuda(model.device),
                    down_projected_original_gradient.flatten().cuda(model.device)
                ).item()

                print(f"Similarity: {similarity}")

                exit(0)


    progress.set_description("Finished calculating flattened similarities")
    return gradient_similarities_random_projected

def calculate_model_generated_gradient_similarities(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    top_k: int = __amount_comparisons
) -> dict[str, dict[str, float]]:
    gradient_similarities: dict[str, dict[str, float]] = {}

    bm25 = build_bm25_index(original_dataset_tokenized)
    paraphrased_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

    progress = tqdm(paraphrased_dataset, desc="Gradients + similarities (model-generated)")

    for paraphrased_sample in progress:
        paraphrased_id = paraphrased_sample["id"]
        # 1) Compute paraphrased gradients with model generation
        paraphrased_grad = get_paraphrased_sample_gradients(
            paraphrased_sample,
            model,
            tokenizer=tokenizer,
            is_model_generated=True,
            paraphrased_config=paraphrased_config
        )
        # 2) BM25 top matches
        paraphrased_text = paraphrased_sample["paraphrased_messages"][0]["content"]
        top_indices = select_top_bm25_matches(
            paraphrased_text,
            bm25,
            original_dataset_tokenized,
            paraphrased_id,
            top_k
        )
        # 3) Loop over top matches and compute similarity
        gradient_similarities[paraphrased_id] = {}
        for original_sample in original_dataset_tokenized.select(top_indices):
            original_id = original_sample["id"]

            progress.set_description(f"P({paraphrased_id}) vs O({original_id})")

            original_grad = get_gradients(model, original_sample)
            sim = compute_flattened_similarity(
                paraphrased_grad, original_grad
            )
            gradient_similarities[paraphrased_id][original_id] = sim

    progress.set_description("Finished model-generated similarities")
    return gradient_similarities


def calculate_paraphrased_layer_dot_products(
        original_dataset_tokenized: Dataset,
        paraphrased_dataset_tokenized: Dataset,
        model: PreTrainedModel,
        top_k: int = __amount_comparisons
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]]
]:
    bm25 = build_bm25_index(original_dataset_tokenized)

    # For final return
    dot_products = {}
    paraphrased_dot_products = {}
    original_dot_products = {}

    progress = tqdm(paraphrased_dataset_tokenized, desc="Layer dot products")

    for paraphrased_sample in progress:
        paraphrased_id = paraphrased_sample["id"]
        # 1) Paraphrased gradients
        paraphrased_grad = get_paraphrased_sample_gradients(
            paraphrased_sample, model, is_model_generated=False
        )
        # 2) Self dot-products for paraphrased
        paraphrased_dot_products[paraphrased_id] = compute_layerwise_self_dot_products(paraphrased_grad, device=model.device)

        # 3) BM25 top matches
        paraphrased_text = paraphrased_sample["paraphrased_messages"][0]["content"]
        top_indices = select_top_bm25_matches(
            paraphrased_text, bm25, original_dataset_tokenized, paraphrased_id, top_k
        )

        dot_products[paraphrased_id] = {}
        for original_sample in original_dataset_tokenized.select(top_indices):
            original_id = original_sample["id"]
            progress.set_description(f"P({paraphrased_id}) vs O({original_id})")

            # 4) Original gradients
            original_grad = get_gradients(model, original_sample)

            # 5) Cache original self-dot-products if not done already
            if original_id not in original_dot_products:
                original_dot_products[original_id] = compute_layerwise_self_dot_products(
                    original_grad, device=model.device
                )

            # 6) Cross dot-products
            dot_products[paraphrased_id][original_id] = compute_layerwise_dot_products(
                paraphrased_grad, original_grad, device=model.device
            )

    return dot_products, paraphrased_dot_products, original_dot_products


def calculate_model_generated_layer_dot_products(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset: Dataset,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    top_k: int = __amount_comparisons
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]]
]:
    dot_products = {}
    paraphrased_dot_products = {}
    original_dot_products = {}

    bm25 = build_bm25_index(original_dataset_tokenized)
    paraphrased_config = get_dataset_config(model, sft_messages_key="paraphrased_messages")

    progress = tqdm(paraphrased_dataset, desc="Model-generated layer dot products")

    for paraphrased_sample in progress:
        paraphrased_id = paraphrased_sample["id"]
        # 1) Paraphrased gradients (model-generated)
        paraphrased_grad = get_paraphrased_sample_gradients(
            paraphrased_sample,
            model,
            tokenizer=tokenizer,
            is_model_generated=True,
            paraphrased_config=paraphrased_config
        )
        # 2) Self dot-products for paraphrased
        paraphrased_dot_products[paraphrased_id] = compute_layerwise_self_dot_products(paraphrased_grad, device=model.device)

        # 3) BM25 top matches
        paraphrased_text = paraphrased_sample["paraphrased_messages"][0]["content"]
        top_indices = select_top_bm25_matches(
            paraphrased_text, bm25, original_dataset_tokenized, paraphrased_id, top_k
        )

        dot_products[paraphrased_id] = {}
        for original_sample in original_dataset_tokenized.select(top_indices):
            original_id = original_sample["id"]
            progress.set_description(f"P({paraphrased_id}) vs O({original_id})")

            # 4) Original gradients
            original_grad = get_gradients(model, original_sample)

            # 5) Cache original self-dot-products if not done already
            if original_id not in original_dot_products:
                original_dot_products[original_id] = compute_layerwise_self_dot_products(
                    original_grad, device=model.device
                )

            # 6) Cross dot-products
            dot_products[paraphrased_id][original_id] = compute_layerwise_dot_products(
                paraphrased_grad, original_grad, device=model.device
            )

    progress.set_description("Finished model-generated layer dot products")
    return dot_products, paraphrased_dot_products, original_dot_products
