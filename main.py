import argparse
import enum
import json
import random
import time

import numpy as np
import torch
import trak
import os

from src.model import get_model, get_tokenizer

from src.computation import (
    calculate_paraphrased_layer_dot_products,
    calculate_model_generated_layer_dot_products,
    calculate_model_generated_gradient_similarities,
    calculate_paraphrased_gradient_similarities,
    calculate_paraphrased_random_projected_gradient_similarities,
    calculate_model_generated_random_projected_gradient_similarities
)
from src.dataset import (
    get_tokenized_datasets,
    get_original_dataset_tokenized,
    get_paraphrased_dataset
)
from src.storage import (
    get_dot_product_paraphrased_file_path,
    get_dot_product_model_generated_file_path,
    get_gradient_similarity_model_generated_file_path,
    get_gradient_similarity_paraphrased_file_path,
    get_gradient_similarity_paraphrased_random_projection_file_path,
    get_gradient_similarity_model_generated_random_projection_file_path
)

class Setting(enum.Enum):
    MODEL_GENERATED = "model-generated"
    PARAPHRASED = "paraphrased"

    def __str__(self):
        return self.value


class ComputationType(enum.Enum):
    DOT_PRODUCT = "dot-product"
    GRADIENT_SIMILARITY = "gradient-similarity"

    def __str__(self):
        return self.value


def parse_args():
    parser = argparse.ArgumentParser(description="Execution of gradient similarity and dot product calculations.")

    parser.add_argument(
        "--setting",
        type=str,
        choices=[setting.value for setting in Setting],
        required=True,
        help="Specify the setting for the computation: model_generated or paraphrased."
    )

    parser.add_argument(
        "--computation-type",
        type=str,
        choices=[computation_type.value for computation_type in ComputationType],
        required=True,
        help="Specify the computation type: dot_product or gradient_similarity."
    )

    parser.add_argument(
        "--use-random-projection",
        action="store_true",
        help="Use random projection for the computation. Only relevant for gradient similarity."
    )

    parser.add_argument(
        "--partition-start",
        type=int,
        default=None,
        help="Start index of the dataset partition to process"
    )

    parser.add_argument(
        "--partition-end",
        type=int,
        default=None,
        help="End index of the dataset partition to process"
    )

    return parser.parse_args()

def main():
    args = parse_args()

    setting = Setting(args.setting)
    computation_type = ComputationType(args.computation_type)
    use_random_projection = args.use_random_projection
    partition_start = args.partition_start
    partition_end = args.partition_end

    print(f"Setting: {setting}")
    print(f"Computation Type: {computation_type}")
    print(f"Use Random Projection: {use_random_projection}")
    if partition_start is not None and partition_end is not None:
        print(f"Processing partition: {partition_start}-{partition_end}")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model = get_model()
    tokenizer = get_tokenizer()

    # start time
    start_time = time.time()

    if setting == Setting.PARAPHRASED:
        if computation_type == ComputationType.DOT_PRODUCT:
            # dot product paraphrased
            original_dataset_tokenized, paraphrased_dataset_tokenized = get_tokenized_datasets(
                model, tokenizer, partition_start, partition_end)

            dot_products, paraphrased_dot_products, original_dot_products = calculate_paraphrased_layer_dot_products(
                original_dataset_tokenized, paraphrased_dataset_tokenized, model)

            # Generate partition-specific filenames if processing a partition
            filename_suffix = ""
            if partition_start is not None and partition_end is not None:
                filename_suffix = f"_part_{partition_start}_{partition_end}"

            # store dot products
            with open(get_dot_product_paraphrased_file_path(f"dot_products{filename_suffix}"), "w") as output_file:
                json.dump(dot_products, output_file, indent=4)

            with open(get_dot_product_paraphrased_file_path(f"paraphrased_dot_products{filename_suffix}"), "w") as output_file:
                json.dump(paraphrased_dot_products, output_file, indent=4)

            with open(get_dot_product_paraphrased_file_path(f"original_dot_products{filename_suffix}"), "w") as output_file:
                json.dump(original_dot_products, output_file, indent=4)

        elif computation_type == ComputationType.GRADIENT_SIMILARITY:
            # gradient similarity paraphrased
            original_dataset_tokenized, paraphrased_dataset_tokenized = get_tokenized_datasets(
                model, tokenizer, partition_start, partition_end)

            # Generate partition-specific filenames if processing a partition
            filename_suffix = ""
            if partition_start is not None and partition_end is not None:
                filename_suffix = f"_part_{partition_start}_{partition_end}"

            if use_random_projection:
                trak.test_install(use_fast_jl=True)

                # random projection
                gradient_similarities = calculate_paraphrased_random_projected_gradient_similarities(
                    original_dataset_tokenized,
                    paraphrased_dataset_tokenized,
                    model
                )

                output_path = get_gradient_similarity_paraphrased_random_projection_file_path()
                if filename_suffix:
                    # Insert suffix before file extension
                    base, ext = os.path.splitext(output_path)
                    output_path = f"{base}{filename_suffix}{ext}"

                with open(output_path, "w") as output_file:
                    json.dump(gradient_similarities, output_file, indent=4)
            else:
                # no random projection
                gradient_similarities = calculate_paraphrased_gradient_similarities(
                    original_dataset_tokenized,
                    paraphrased_dataset_tokenized,
                    model
                )

                output_path = get_gradient_similarity_paraphrased_file_path()
                if filename_suffix:
                    # Insert suffix before file extension
                    base, ext = os.path.splitext(output_path)
                    output_path = f"{base}{filename_suffix}{ext}"

                with open(output_path, "w") as output_file:
                    json.dump(gradient_similarities, output_file, indent=4)

    elif setting == Setting.MODEL_GENERATED:
        if computation_type == ComputationType.DOT_PRODUCT:
            # dot product model generated
            original_dataset_tokenized, paraphrased_dataset_tokenized = get_tokenized_datasets(
                model, tokenizer, partition_start, partition_end)

            dot_products, paraphrased_dot_products, original_dot_products = calculate_model_generated_layer_dot_products(
                original_dataset_tokenized, paraphrased_dataset_tokenized, model, tokenizer)

            # Generate partition-specific filenames if processing a partition
            filename_suffix = ""
            if partition_start is not None and partition_end is not None:
                filename_suffix = f"_part_{partition_start}_{partition_end}"

            # store dot products
            with open(get_dot_product_model_generated_file_path(f"dot_products{filename_suffix}"), "w") as output_file:
                json.dump(dot_products, output_file, indent=4)

            with open(get_dot_product_model_generated_file_path(f"paraphrased_dot_products{filename_suffix}"), "w") as output_file:
                json.dump(paraphrased_dot_products, output_file, indent=4)

            with open(get_dot_product_model_generated_file_path(f"original_dot_products{filename_suffix}"), "w") as output_file:
                json.dump(original_dot_products, output_file, indent=4)

        elif computation_type == ComputationType.GRADIENT_SIMILARITY:
            # gradient similarity model generated
            original_dataset_tokenized = get_original_dataset_tokenized(model, tokenizer)
            paraphrased_dataset = get_paraphrased_dataset(partition_start=partition_start, partition_end=partition_end)

            # Generate partition-specific filenames if processing a partition
            filename_suffix = ""
            if partition_start is not None and partition_end is not None:
                filename_suffix = f"_part_{partition_start}_{partition_end}"

            if use_random_projection:
                trak.test_install(use_fast_jl=True)

                # random projection
                gradient_similarities = calculate_model_generated_random_projected_gradient_similarities(
                    original_dataset_tokenized,
                    paraphrased_dataset,
                    model,
                    tokenizer
                )

                output_path = get_gradient_similarity_model_generated_random_projection_file_path()
                if filename_suffix:
                    # Insert suffix before file extension
                    base, ext = os.path.splitext(output_path)
                    output_path = f"{base}{filename_suffix}{ext}"

                with open(output_path, "w") as output_file:
                    json.dump(gradient_similarities, output_file, indent=4)
            else:
                # no random projection
                gradient_similarities = calculate_model_generated_gradient_similarities(
                    original_dataset_tokenized,
                    paraphrased_dataset,
                    model,
                    tokenizer
                )

                output_path = get_gradient_similarity_model_generated_file_path()
                if filename_suffix:
                    # Insert suffix before file extension
                    base, ext = os.path.splitext(output_path)
                    output_path = f"{base}{filename_suffix}{ext}"

                with open(output_path, "w") as output_file:
                    json.dump(gradient_similarities, output_file, indent=4)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == '__main__':
    main()
