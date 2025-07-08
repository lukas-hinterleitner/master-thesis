import datasets
import numpy as np
import torch
from datasets import Dataset
from rank_bm25 import BM25Okapi
from torch.nn import CosineSimilarity
from tqdm import tqdm
from trak.projectors import CudaProjector, ProjectionType
from transformers import PreTrainedModel

# Local project utilities ----------------------------------------------------
from src.model import get_num_parameters_per_layer
from src.model_operations import (
    get_gradients,
    get_flattened_weight_vector,
)

# Disable noisy HF progress bars globally
datasets.disable_progress_bar()

# ============================================================================
# Constants & simple helpers
# ============================================================================

__amount_comparisons = 5  # default number of originals to compare per sample
__projection_percents = (0.01, 0.05)  # 1% and 5% random‑projection sizes


def __simple_tokenize(doc: str):
    """
    Splits the input string into a list of individual words or tokens.

    This function takes a single string as input and splits it into
    a list of tokens based on spaces. It does not handle complex
    tokenization or punctuation removal, providing only a basic
    space-based split of the input string.

    :param doc: The input string that will be tokenized.
    :type doc: str
    :return: A list of tokens obtained by splitting the input string.
    :rtype: list[str]
    """
    return doc.split()


# ---------------------------------------------------------------------------
# Layer-wise projection helpers
# ---------------------------------------------------------------------------

def __project_gradients_layerwise(
    gradients: dict[str, torch.Tensor],
    layer_proj_dims: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """
    Projects gradients layer-wise using CudaProjector and concatenates the results.

    :param gradients: Dictionary mapping layer names to gradient tensors
    :param layer_proj_dims: Dictionary mapping layer names to projection dimensions
    :param device: Device to perform computations on
    :return: Concatenated projected gradients tensor
    """
    projected_parts = []

    for layer_name, layer_grad in gradients.items():
        if layer_name not in layer_proj_dims:
            continue

        # Create layer-specific projector
        layer_projector = CudaProjector(
            grad_dim=layer_grad.numel(),
            proj_dim=layer_proj_dims[layer_name],
            seed=42 + hash(layer_name) % 10000,  # Different seed per layer
            device=device,
            proj_type=ProjectionType.rademacher,
            max_batch_size=8,
        )

        # Project and collect
        flat_layer = layer_grad.flatten().half()
        projected = layer_projector.project(
            grads=flat_layer.reshape(1, -1).cuda(device),
            model_id=0
        ).cpu()

        projected_parts.append(projected.flatten())

        # Clear CUDA cache after each layer
        torch.cuda.empty_cache()

    # Concatenate all layer projections
    return torch.cat(projected_parts)


def __calculate_layer_projection_dimensions(
    model: PreTrainedModel,
    proj_dim: int,
) -> dict[str, int]:
    """
    Calculate proportional projection dimensions for each layer.

    :param model: The pre-trained model
    :param proj_dim: Total projection dimension
    :return: Dictionary mapping layer names to their projection dimensions
    """
    layer_params = get_num_parameters_per_layer(model)
    total_params = model.num_parameters()

    return {
        layer: max(512, int(512 * round((count / total_params * proj_dim) / 512)))
        for layer, count in layer_params.items()
    }


# ---------------------------------------------------------------------------
# BM25 helpers
# ---------------------------------------------------------------------------

def build_bm25_index(original_dataset_tokenized: Dataset) -> BM25Okapi:
    """Pre‑compute a BM25 index over the *first* user message of each sample."""
    original_texts = [row["messages"][0]["content"] for row in original_dataset_tokenized]
    tokenized_corpus = [__simple_tokenize(doc) for doc in original_texts]
    return BM25Okapi(tokenized_corpus)


def select_top_bm25_matches(
    query_text: str,
    bm25: BM25Okapi,
    original_dataset_tokenized: Dataset,
    paraphrased_id: str,
    top_k: int,
) -> list[int]:
    """
    Selects the top matches for a query using BM25 scores.

    This function takes a query text, runs it through a BM25 model, and retrieves
    the top `k` matches from the dataset based on cosine similarity scores. If the
    ground-truth original identifier (`paraphrased_id`) corresponding to the
    query text exists in the dataset, it ensures that it is always part of the
    returned top candidates, replacing the least relevant result if necessary.

    :param query_text: The input query text to find matches for.
    :param bm25: The BM25 model instance used to compute similarity scores.
    :param original_dataset_tokenized: The tokenized dataset that the function
        searches over to find the best matches.
    :param paraphrased_id: The unique identifier of the ground-truth item in the
        dataset to ensure inclusion in the top results.
    :param top_k: The number of top matches to retrieve from the dataset.
    :return: A list of indices representing the top `k` matches in the dataset.
    """

    scores = bm25.get_scores(__simple_tokenize(query_text))
    top_indices = np.argsort(-scores)[:top_k]  # highest → lowest

    # Ensure ground‑truth original (by ID) is always among candidates
    if paraphrased_id in original_dataset_tokenized["id"]:
        matching_original_idx = original_dataset_tokenized["id"].index(paraphrased_id)
        if matching_original_idx not in top_indices:
            top_indices[-1] = matching_original_idx  # replace worst with truth

    return top_indices.tolist()

# ---------------------------------------------------------------------------
# Random‑projection helpers
# ---------------------------------------------------------------------------

def __projection_dimensions(model: PreTrainedModel) -> list[int]:
    """
    Calculates the projection dimensions for a given pre-trained model based on the
    pre-specified projection percentages. The dimensions are adjusted to the nearest
    multiple of 512 for compatibility or efficiency considerations.

    :param model: The pre-trained model whose projection dimensions need to be
        calculated.
    :type model: PreTrainedModel

    :return: A list of integers representing the calculated projection dimensions for
        the given model.
    :rtype: list[int]
    """

    dims = [int(model.num_parameters() * p) for p in __projection_percents]
    return [int(round(dim / 512) * 512) for dim in dims]


# ============================================================================
# Unified *internal* pipelines (private). Public wrappers simply forward.
# ============================================================================


def __calculate_flattened_similarities(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset_tokenized: Dataset,
    model: PreTrainedModel,
    top_k: int,
    is_model_generated: bool,
) -> dict[str, dict[str, float]]:
    """
    Calculate similarities between paraphrased datasets and original datasets,
    utilizing gradients and selected top matches. This function uses a cosine
    similarity measure and BM25 for candidate selection. The process involves
    computing gradients for paraphrased samples, retrieving top-matching
    originals, and comparing their gradients for similarity.

    :param original_dataset_tokenized: The tokenized dataset containing original data samples.
    :param paraphrased_dataset_tokenized: An iterable or dataset containing paraphrased data samples.
    :param model: A pre-trained model that processes the input datasets and computes gradients.
    :param top_k: The number of top similar candidates to retrieve for comparison using BM25.
    :param is_model_generated: A boolean specifying whether the paraphrased dataset is generated
        by the model.
    :return: A dictionary where each key is a paraphrased sample ID, and its value is another dictionary
        mapping original sample IDs to their calculated gradient similarity scores.
    """

    messages_key = "model_generated_messages" if is_model_generated else "paraphrased_messages"

    similarity_function = CosineSimilarity(dim=0)
    bm25 = build_bm25_index(original_dataset_tokenized)

    gradient_similarities: dict[str, dict[str, float]] = {}

    progress = tqdm(
        paraphrased_dataset_tokenized,
        desc="Gradients + flattened similarities%s"
        % (" (model‑generated)" if is_model_generated else ""),
    )

    for paraphrased_sample in progress:
        paraphrased_id = paraphrased_sample["id"]

        # 1) Paraphrased gradients ------------------------------------------------
        paraphrased_grad = get_gradients(model, paraphrased_sample)

        # 2) Candidate originals via BM25 ---------------------------------------
        paraphrased_text = paraphrased_sample[messages_key][0]["content"]
        top_indices = select_top_bm25_matches(
            paraphrased_text, bm25, original_dataset_tokenized, paraphrased_id, top_k
        )

        # 3) Compare gradients ---------------------------------------------------
        gradient_similarities[paraphrased_id] = {}
        for original_sample in original_dataset_tokenized.select(top_indices):
            original_id = original_sample["id"]
            progress.set_description(f"P({paraphrased_id}) vs O({original_id})")

            original_grad = get_gradients(model, original_sample)
            grad_sim = similarity_function(
                get_flattened_weight_vector(paraphrased_grad).to(model.device),
                get_flattened_weight_vector(original_grad).to(model.device),
            ).item()
            gradient_similarities[paraphrased_id][original_id] = grad_sim

    progress.set_description("Finished flattened similarities")
    return gradient_similarities


# ---------------------------------------------------------------------------

def __calculate_random_projected_similarities(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset_tokenized: Dataset,
    model: PreTrainedModel,
    top_k: int,
    is_model_generated: bool,
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Calculates random-projected similarities between paraphrased datasets and the original
    dataset, leveraging gradient projection. This function uses cosine similarity to
    evaluate the relationships at different projection dimensions.

    :param original_dataset_tokenized: The original dataset post tokenization.
    :param paraphrased_dataset_tokenized: An iterable or dataset containing the paraphrased samples.
    :param model: The pre-trained model used for gradient computations and projections.
    :param top_k: The maximum number of candidates selected using BM25 for comparisons.
    :param is_model_generated: Specifies whether the paraphrased dataset is model-generated.
    :return: A dictionary where keys are projection dimensions and values are nested
        dictionaries mapping paraphrased sample IDs to original sample IDs along with
        their similarity scores.
    """

    messages_key = "model_generated_messages" if is_model_generated else "paraphrased_messages"

    bm25 = build_bm25_index(original_dataset_tokenized)
    similarity_function = CosineSimilarity(dim=0)

    results: dict[int, dict[str, dict[str, float]]] = {}

    for proj_dim in tqdm(__projection_dimensions(model), desc="Projection dimensions", position=0):
        print(f"Projection dimension: {proj_dim}")
        results[proj_dim] = {}

        # Calculate layer-wise projection dimensions proportionally
        layer_proj_dims = __calculate_layer_projection_dimensions(model, proj_dim)

        progress = tqdm(
            paraphrased_dataset_tokenized,
            desc="Gradients + similarities (layer-wise projection%s)"
                 % (" (model‑generated)" if is_model_generated else ""),
            position=1,
            leave=False,
        )

        for paraphrased_sample in progress:
            paraphrased_id = paraphrased_sample["id"]

            # 1) Paraphrased gradients
            paraphrased_grad = get_gradients(model, paraphrased_sample)

            # Project paraphrased gradients layer-wise
            down_projected_paraphrased = __project_gradients_layerwise(paraphrased_grad, layer_proj_dims, model.device)

            # 2) Candidates via BM25
            paraphrased_text = paraphrased_sample[messages_key][0]["content"]
            top_indices = select_top_bm25_matches(
                paraphrased_text, bm25, original_dataset_tokenized, paraphrased_id, top_k
            )

            # 3) Compare to originals
            results[proj_dim][paraphrased_id] = {}
            for original_sample in original_dataset_tokenized.select(top_indices):
                original_id = original_sample["id"]
                progress.set_description(f"P({paraphrased_id}) vs O({original_id})")

                original_grad = get_gradients(model, original_sample)

                # Project original gradients layer-wise
                down_projected_original = __project_gradients_layerwise(original_grad, layer_proj_dims, model.device)

                # Calculate similarity
                sim = similarity_function(
                    down_projected_paraphrased.cuda(model.device),
                    down_projected_original.cuda(model.device),
                ).item()
                results[proj_dim][paraphrased_id][original_id] = sim

                # Clear CUDA cache after each comparison
                torch.cuda.empty_cache()

        progress.set_description("Finished layer-wise random‑projected similarities")

    return results


# ---------------------------------------------------------------------------


def __calculate_layerwise_dot_products(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset_tokenized: Dataset,
    model: PreTrainedModel,
    top_k: int,
    is_model_generated: bool,
) -> tuple[
    dict[str, dict[str, dict[str, float]]],
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    """
    Calculates layer-wise dot products for paraphrased and original datasets using gradients,
    providing additional data on self-dot products for both paraphrased and original
    samples. Utilizes BM25 index for selecting top-k matches from the original dataset to
    compare against paraphrased samples.

    :param original_dataset_tokenized: Dataset object containing the tokenized original samples.
    :param paraphrased_dataset_tokenized: An iterable of paraphrased dataset, which may be either a
        regular Dataset or an IterableDataset.
    :param model: Pre-trained model instance for generating gradients and other computations.
    :param top_k: Number of top BM25 matches to retrieve for comparing original samples with
        paraphrased ones.
    :param is_model_generated: Boolean indicating whether the paraphrased dataset is generated
        using the model or not.
    :return: A tuple containing three dictionaries:
        1. Dictionary with layer-wise dot products for paraphrased vs. original samples.
        2. Dictionary with layer-wise self-dot products for paraphrased samples.
        3. Dictionary with layer-wise self-dot products for original samples.
    """

    messages_key = "model_generated_messages" if is_model_generated else "paraphrased_messages"

    bm25 = build_bm25_index(original_dataset_tokenized)

    dot_products: dict[str, dict[str, dict[str, float]]] = {}
    paraphrased_self: dict[str, dict[str, float]] = {}
    original_self: dict[str, dict[str, float]] = {}

    progress = tqdm(
        paraphrased_dataset_tokenized,
        desc="Layer dot products%s" % (" (model‑generated)" if is_model_generated else ""),
    )

    for paraphrased_sample in progress:
        paraphrased_id = paraphrased_sample["id"]

        # 1) Paraphrased gradients & self‑dot ------------------------------------
        paraphrased_grad = get_gradients(model, paraphrased_sample)
        paraphrased_self[paraphrased_id] = __compute_layerwise_self_dot_products(
            paraphrased_grad, device=model.device
        )

        # 2) BM25 candidates -----------------------------------------------------
        paraphrased_text = paraphrased_sample[messages_key][0]["content"]
        top_indices = select_top_bm25_matches(
            paraphrased_text, bm25, original_dataset_tokenized, paraphrased_id, top_k
        )

        # 3) Cross dot products --------------------------------------------------
        dot_products[paraphrased_id] = {}
        for original_sample in original_dataset_tokenized.select(top_indices):
            original_id = original_sample["id"]
            progress.set_description(f"P({paraphrased_id}) vs O({original_id})")

            original_grad = get_gradients(model, original_sample)
            if original_id not in original_self:
                original_self[original_id] = __compute_layerwise_self_dot_products(
                    original_grad, device=model.device
                )

            dot_products[paraphrased_id][original_id] = __compute_layerwise_dot_products(
                paraphrased_grad, original_grad, device=model.device
            )

    progress.set_description("Finished layer‑wise dot products")
    return dot_products, paraphrased_self, original_self


# --- Flattened similarities -------------------------------------------------

def calculate_paraphrased_gradient_similarities(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset_tokenized: Dataset,
    model: PreTrainedModel,
    top_k: int = __amount_comparisons,
) -> dict[str, dict[str, float]]:
    """
    Calculate the gradient similarities between the tokens of an original and a paraphrased dataset.

    This function processes tokenized datasets to evaluate the gradient similarities
    between pairs of original and paraphrased data points using a specified
    pre-trained model. It calculates similarities based on model representations,
    and outputs the results formatted as a dictionary. The function also allows control
    over the number of comparisons made through the `top_k` parameter.

    :param original_dataset_tokenized:
        The original dataset, tokenized for processing by the model.
    :param paraphrased_dataset_tokenized:
        The paraphrased dataset, tokenized to match the original format for processing.
    :param model:
        A pre-trained model used for calculating gradient similarities.
    :param top_k:
        The number of data point comparisons to perform, defaulting to a pre-set value
        if not explicitly provided.
    :return:
        A dictionary where keys are identifiers of data points and values are nested
        dictionaries that represent the gradient similarity scores for each paraphrased
        data point compared to the original.
    """

    return __calculate_flattened_similarities(
        original_dataset_tokenized=original_dataset_tokenized,
        paraphrased_dataset_tokenized=paraphrased_dataset_tokenized,
        model=model,
        top_k=top_k,
        is_model_generated=False,
    )


def calculate_model_generated_gradient_similarities(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset: Dataset,
    model: PreTrainedModel,
    top_k: int = __amount_comparisons,
) -> dict[str, dict[str, float]]:
    """
    Calculates the similarities of model-generated gradients between the original
    and paraphrased datasets. This function evaluates the gradient similarities by
    using a pre-trained model and tokenizer and processes up to the specified number
    of comparisons.

    It makes use of a helper function to calculate these similarities and outputs a
    dictionary where the keys correspond to data entries and the values map to their
    similarity scores. This method is particularly useful for tasks related to
    gradient-based evaluations or perturbation analysis.

    :param original_dataset_tokenized: The tokenized form of the original dataset.
    :param paraphrased_dataset: The dataset containing paraphrased entries that are
        used for similarity comparisons.
    :param model: The pre-trained model used for calculating gradient similarities
        between datasets.
    :param top_k: The maximum number of comparisons to process in the calculation.
        Defaults to a pre-configured amount if not provided.
    :return: A nested dictionary where the outer keys are dataset entry identifiers,
        and inner dictionaries map to corresponding similarity scores.
    """

    return __calculate_flattened_similarities(
        original_dataset_tokenized=original_dataset_tokenized,
        paraphrased_dataset_tokenized=paraphrased_dataset,
        model=model,
        top_k=top_k,
        is_model_generated=True,
    )


# --- Random‑projected similarities -----------------------------------------

def calculate_paraphrased_random_projected_gradient_similarities(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset_tokenized: Dataset,
    model: PreTrainedModel,
    top_k: int = __amount_comparisons,
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Calculate similarities between original and paraphrased datasets using random
    projected gradient methodology.

    This function utilizes a pretrained model to compute similarities between
    tokenized datasets. It performs random projections to reduce dimensionality
    and then compares distributions between the original and paraphrased data
    using top-K gradient similarity calculations.

    :param original_dataset_tokenized:
        Tokenized dataset representing the original input data.
    :param paraphrased_dataset_tokenized:
        Tokenized dataset representing the paraphrased input data.
    :param model:
        Pretrained model used for generating gradient similarities.
    :param top_k:
        Number of top-K entities to include in the similarity calculation.
    :return:
        A dictionary mapping each instance to nested dictionaries containing
        the similarity scores between original and paraphrased data.
    """

    return __calculate_random_projected_similarities(
        original_dataset_tokenized=original_dataset_tokenized,
        paraphrased_dataset_tokenized=paraphrased_dataset_tokenized,
        model=model,
        top_k=top_k,
        is_model_generated=False,
    )


def calculate_model_generated_random_projected_gradient_similarities(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset_tokenized: Dataset,
    model: PreTrainedModel,
    top_k: int = __amount_comparisons,
) -> dict[int, dict[str, dict[str, float]]]:
    """
    Calculate the similarities of model-generated data after random projection of gradients.

    This function computes the similarities between the original dataset and
    paraphrased dataset based on random projection of gradients, using a
    specified model and tokenizer. It assesses the degree to which paraphrased
    data retains the original content, leveraging specific top variations and
    model-supplied classifications.

    :param original_dataset_tokenized: Tokenized dataset representing the original data.
    :type original_dataset_tokenized: Dataset
    :param paraphrased_dataset_tokenized: Tokenized dataset representing the paraphrased data.
    :type paraphrased_dataset_tokenized: Dataset
    :param model: Pre-trained model used for gradient computations.
    :type model: PreTrainedModel
    :param top_k: The number of largest similarity scores to evaluate. Defaults to global variable __amount_comparisons.
    :type top_k: int
    :return: A nested dictionary where each key corresponds to unique reference identifiers,
             with values containing inner dictionaries of similarity metrics between the datasets.
    :rtype: dict[int, dict[str, dict[str, float]]]
    """

    return __calculate_random_projected_similarities(
        original_dataset_tokenized=original_dataset_tokenized,
        paraphrased_dataset_tokenized=paraphrased_dataset_tokenized,
        model=model,
        top_k=top_k,
        is_model_generated=True,
    )


# --- Layer‑wise dot products ----------------------------------------------

def calculate_paraphrased_layer_dot_products(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset_tokenized: Dataset,
    model: PreTrainedModel,
    top_k: int = __amount_comparisons,
):
    """
    Calculates the layer-wise dot products between the original dataset and its paraphrased counterpart,
    using a specified model. This function computes dot products for the top_k pairs based on similarity
    of tokens across layers, leveraging a pre-trained model for the embeddings.

    :param original_dataset_tokenized: Tokenized dataset representing the original text.
    :param paraphrased_dataset_tokenized: Tokenized dataset of the paraphrased version of the original text.
    :param model: The pre-trained transformer model used for computing embeddings and similarity.
    :param top_k: Specifies the number of top pairs to consider for calculating dot products, based on
        token similarity.
    :return: List of computed dot products for each layer, comparing original text embeddings to the
        paraphrased text embeddings.
    """

    return __calculate_layerwise_dot_products(
        original_dataset_tokenized=original_dataset_tokenized,
        paraphrased_dataset_tokenized=paraphrased_dataset_tokenized,
        model=model,
        top_k=top_k,
        is_model_generated=False,
    )


def calculate_model_generated_layer_dot_products(
    original_dataset_tokenized: Dataset,
    paraphrased_dataset: Dataset,
    model: PreTrainedModel,
    top_k: int = __amount_comparisons,
):
    """
    Calculate layer-wise dot products for model-generated paraphrased data.

    This function computes dot products between the original and paraphrased datasets
    at each layer of the specified model. The paraphrased dataset is expected to be
    generated by the model itself. This process helps to analyze layer-specific
    behaviors and representations between the original and generated data.

    :param original_dataset_tokenized: Dataset
        The tokenized dataset containing the original input data for analysis.
    :param paraphrased_dataset: Dataset
        The dataset containing paraphrased versions of the inputs, generated by the
        model.
    :param model: PreTrainedModel
        The transformer model used to analyze and compute the representations.
    :param top_k: Int
        The number of top comparisons or dot-product computations to perform for
        evaluation.

    :return: Dict[str, Any]
        A dictionary containing layer-wise dot products and associated metrics
        for the comparison between the original and paraphrased datasets.
    """

    return __calculate_layerwise_dot_products(
        original_dataset_tokenized=original_dataset_tokenized,
        paraphrased_dataset_tokenized=paraphrased_dataset,
        model=model,
        top_k=top_k,
        is_model_generated=True,
    )


# ============================================================================
# Internal math helpers (unchanged logic) – kept at bottom to mirror original
# ============================================================================


def __compute_layerwise_dot_products(
    gradients1: dict[str, torch.Tensor],
    gradients2: dict[str, torch.Tensor],
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """
    Computes dot products for corresponding gradient tensors layer-wise. This utility
    takes two dictionaries containing gradient tensors for different layers and computes
    the dot product for each layer's gradients, ensuring alignment between the two sets
    of gradients. Returns a dictionary mapping layer name to their respective dot product
    values, with computations performed on the specified device.

    :param gradients1: Dictionary mapping layer names to gradient tensors. Represents
                      the first set of gradients.
    :param gradients2: Dictionary mapping layer names to gradient tensors. Represents
                      the second set of gradients.
    :param device: Device on which dot product computations are executed. Defaults to "cpu".
    :return: A dictionary mapping layer names to the corresponding float values of dot
             products between the gradients from `gradients1` and `gradients2`.
    """

    out: dict[str, float] = {}
    for (layer, g1), (_, g2) in zip(gradients1.items(), gradients2.items(), strict=True):
        out[layer] = g1.flatten().to(device).dot(g2.flatten().to(device)).item()
    return out


def __compute_layerwise_self_dot_products(
    gradients: dict[str, torch.Tensor],
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """
    Computes the self-dot product for each layer's gradient, returning a dictionary
    where the keys correspond to layers and the values are the resulting dot
    products.

    :param gradients: Dictionary where keys are layer names (str) and values are
        gradient tensors (torch.Tensor).
    :param device: The device (torch.device) on which the computations will be
        performed. Defaults to CPU.
    :return: A dictionary mapping layer names (str) to their self dot product
        values (float). Each value is computed by flattening the corresponding
        gradient tensor and calculating the dot product with itself.
    """

    out: dict[str, float] = {}
    for layer, grad in gradients.items():
        flat = grad.flatten().to(device)
        out[layer] = flat.dot(flat).item()
    return out
