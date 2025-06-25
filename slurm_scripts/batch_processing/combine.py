import argparse
import json
import os
import sys
from pathlib import Path
from collections import OrderedDict

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.config.model import MODEL_NAME
from src.config.storage import (
    gradient_similarity_paraphrased_storage_path,
    gradient_similarity_model_generated_storage_path,
    gradient_similarity_random_projection_paraphrased_storage_path,
    gradient_similarity_random_projection_model_generated_storage_path,
    dot_product_paraphrased_storage_path,
    dot_product_model_generated_storage_path
)


def find_partial_results(base_path, pattern="*_part_*_*.json"):
    """Find all partial result files matching the pattern"""
    if not os.path.exists(base_path):
        print(f"Path does not exist: {base_path}")
        return []

    # Create a path object for more reliable glob matching
    path = Path(base_path)
    return list(path.glob(pattern))


def combine_gradient_similarity_results(file_paths, output_path):
    """Combine gradient similarity results from multiple files"""
    combined_results = {}

    for file_path in file_paths:
        with open(file_path, 'r') as f:
            partial_results = json.load(f)

        # For standard gradient similarity results
        if isinstance(partial_results, dict) and not any(isinstance(v, dict) for v in partial_results.values()):
            combined_results.update(partial_results)

        # For nested results (like random projection results with dimensions as keys)
        elif isinstance(partial_results, dict):
            for dimension, results in partial_results.items():
                if dimension not in combined_results:
                    combined_results[dimension] = {}
                combined_results[dimension].update(results)

    # Sort the combined results by their first key
    if combined_results:
        # Check if it's nested (random projection results) or flat structure
        first_key = next(iter(combined_results))
        if isinstance(combined_results[first_key], dict):
            # Nested structure - sort each dimension's results by key
            sorted_results = OrderedDict()
            for dimension in sorted(combined_results.keys()):
                sorted_results[dimension] = OrderedDict(sorted(combined_results[dimension].items()))
            combined_results = sorted_results
        else:
            # Flat structure - sort by main keys
            combined_results = OrderedDict(sorted(combined_results.items()))

    with open(output_path, 'w') as f:
        json.dump(combined_results, f, indent=4)

    print(f"Combined results written to {output_path}")


def combine_dot_product_results(file_paths, output_path_template):
    """Combine dot product results from multiple files"""
    combined_dot_products = {}
    combined_paraphrased_dot_products = {}
    combined_original_dot_products = {}

    # Process dot_products files
    for file_path in [p for p in file_paths if "dot_products_part" in p.name and "paraphrased_dot_products_part" not in p.name and "original_dot_products_part" not in p.name]:
        with open(file_path, 'r') as f:
            results = json.load(f)
            for paraphrased_id, original_comparisons in results.items():
                if paraphrased_id not in combined_dot_products:
                    combined_dot_products[paraphrased_id] = {}

                for original_id, layer_dot_product_comparisons in original_comparisons.items():
                    if original_id not in combined_dot_products[paraphrased_id]:
                        combined_dot_products[paraphrased_id][original_id] = {}

                    combined_dot_products[paraphrased_id][original_id].update(layer_dot_product_comparisons)

    # Process paraphrased_dot_products files
    for file_path in [p for p in file_paths if "paraphrased_dot_products_part" in p.name]:
        with open(file_path, 'r') as f:
            results = json.load(f)
            combined_paraphrased_dot_products.update(results)

    # Process original_dot_products files
    for file_path in [p for p in file_paths if "original_dot_products_part" in p.name]:
        with open(file_path, 'r') as f:
            results = json.load(f)
            combined_original_dot_products.update(results)

    # Sort all results by their keys
    combined_dot_products = OrderedDict(sorted(combined_dot_products.items()))
    combined_paraphrased_dot_products = OrderedDict(sorted(combined_paraphrased_dot_products.items()))
    combined_original_dot_products = OrderedDict(sorted(combined_original_dot_products.items()))

    # Write combined results
    with open(output_path_template.format("dot_products"), 'w') as f:
        json.dump(combined_dot_products, f, indent=4)

    with open(output_path_template.format("paraphrased_dot_products"), 'w') as f:
        json.dump(combined_paraphrased_dot_products, f, indent=4)

    with open(output_path_template.format("original_dot_products"), 'w') as f:
        json.dump(combined_original_dot_products, f, indent=4)

    print(f"Combined dot product results written to {output_path_template.format('*')}")


def main():
    parser = argparse.ArgumentParser(description="Combine partial results from distributed processing")

    parser.add_argument(
        "--result-type",
        type=str,
        choices=[
            "paraphrased-gradient-similarities",
            "paraphrased-random-projected-gradient-similarities",
            "paraphrased-layer-dot-products",
            "model-generated-gradient-similarities",
            "model-generated-random-projected-gradient-similarities",
            "model-generated-layer-dot-products"
        ],
        required=True,
        help="Type of results to combine"
    )

    args = parser.parse_args()

    # Get model name and sample size from config
    model_name = MODEL_NAME

    # Determine paths based on result type
    if args.result_type == "paraphrased-gradient-similarities":
        base_path = os.path.join(gradient_similarity_paraphrased_storage_path, model_name)
        file_pattern = "sample_size_*_part_*_*.json"
        output_path = os.path.join(base_path, "sample_size_full.json")
        combine_function = combine_gradient_similarity_results

    elif args.result_type == "paraphrased-random-projected-gradient-similarities":
        base_path = os.path.join(gradient_similarity_random_projection_paraphrased_storage_path, model_name)
        file_pattern = "sample_size_*_part_*_*.json"
        output_path = os.path.join(base_path, "sample_size_full.json")
        combine_function = combine_gradient_similarity_results

    elif args.result_type == "paraphrased-layer-dot-products":
        base_path = os.path.join(dot_product_paraphrased_storage_path, model_name, "sample_size", "full")
        os.makedirs(base_path, exist_ok=True)
        file_pattern = "*_part_*_*.json"
        output_path_template = os.path.join(base_path, "{}.json")
        # Search in parent directories for the partial results
        search_path = Path(dot_product_paraphrased_storage_path) / model_name
        file_paths = list(search_path.glob(f"**/{file_pattern}"))
        combine_dot_product_results(file_paths, output_path_template)
        return

    elif args.result_type == "model-generated-gradient-similarities":
        base_path = os.path.join(gradient_similarity_model_generated_storage_path, model_name)
        file_pattern = "sample_size_*_part_*_*.json"
        output_path = os.path.join(base_path, "sample_size_full.json")
        combine_function = combine_gradient_similarity_results

    elif args.result_type == "model-generated-random-projected-gradient-similarities":
        base_path = os.path.join(gradient_similarity_random_projection_model_generated_storage_path, model_name)
        file_pattern = "sample_size_*_part_*_*.json"
        output_path = os.path.join(base_path, "sample_size_full.json")
        combine_function = combine_gradient_similarity_results

    elif args.result_type == "model-generated-layer-dot-products":
        base_path = os.path.join(dot_product_model_generated_storage_path, model_name, "sample_size", "full")
        os.makedirs(base_path, exist_ok=True)
        file_pattern = "*_part_*_*.json"
        output_path_template = os.path.join(base_path, "{}.json")
        # Search in parent directories for the partial results
        search_path = Path(dot_product_model_generated_storage_path) / model_name
        file_paths = list(search_path.glob(f"**/{file_pattern}"))
        combine_dot_product_results(file_paths, output_path_template)
        return

    # Find partial result files
    file_paths = find_partial_results(base_path, file_pattern)

    if not file_paths:
        print(f"No partial result files found in {base_path} matching pattern {file_pattern}")
        return

    print(f"Found {len(file_paths)} partial result files")

    # Combine results
    if args.result_type in ["paraphrased-layer-dot-products", "model-generated-layer-dot-products"]:
        combine_dot_product_results(file_paths, output_path)
    else:
        combine_gradient_similarity_results(file_paths, output_path)


if __name__ == "__main__":
    main()