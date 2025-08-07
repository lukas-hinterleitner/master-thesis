import argparse
import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from src.dataset import get_original_dataset_tokenized
from src.model_operations import get_gradients
from src.computation import __projection_dimensions
from trak.projectors import CudaProjector, ProjectionType
from src.model import get_model, get_tokenizer


def compute_full_gradient_projection(
    model: PreTrainedModel,
    sample: dict,
    proj_dim: int,
) -> torch.Tensor:
    """
    Computes the random projection for the full gradient of a single sample.

    :param model: The pre-trained model.
    :param sample: The input sample (a dictionary).
    :param device: The device to perform computations on.
    :param proj_dim: The dimension of the projection.
    :return: The projected gradient tensor.
    """
    # 1. Get gradients for the sample
    gradients = get_gradients(model, sample)

    # 2. Flatten all gradients into a single vector
    full_gradient_vector = torch.cat([g.flatten() for g in gradients.values()])

    print(full_gradient_vector.shape)

    # 3. Create a projector for the full gradient
    projector = CudaProjector(
        grad_dim=full_gradient_vector.numel(),
        proj_dim=proj_dim,
        seed=42,
        device=model.device,
        proj_type=ProjectionType.rademacher,
        max_batch_size=1
    )

    # 4. Project the full gradient
    projected_gradient = projector.project(
        grads=full_gradient_vector.reshape(1, -1).cuda(model.device),
        model_id=0
    ).cpu()

    return projected_gradient.flatten()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute random projection for a single sample on the full gradient.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model to use.")

    args = parser.parse_args()

    # Load model and tokenizer
    model = get_model(args.model_name)
    tokenizer = get_tokenizer(args.model_name)

    # Load dataset
    sample = get_original_dataset_tokenized(model, tokenizer)[0]

    # Calculate projection dimension
    total_params = model.num_parameters()
    proj_dim = int(total_params * 0.01)
    proj_dim = int(round(proj_dim / 512) * 512) # Adjust to nearest multiple of 512

    # Compute projection
    projected_gradient = compute_full_gradient_projection(model, sample, proj_dim)

    print(f"Projected gradient for the first sample:")
    print(projected_gradient)
    print(f"Shape of projected gradient: {projected_gradient.shape}")
