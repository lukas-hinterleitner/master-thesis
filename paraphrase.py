import argparse
import random

import numpy as np
import torch

from datasets import load_dataset, Dataset

from src.dataset import get_paraphrased_dataset_tokenized
from src.model_operations import generate_model_output_from_paraphrased_sample, map_to_message_format
from src.model import get_model, get_tokenizer

from tqdm import tqdm

from src.storage import get_model_generated_huggingface_dataset_path

def create_paraphrased_dataset():
    from src.paraphrasing import paraphrase_input

    """Create paraphrased dataset from LIMA data."""
    data = load_dataset("allenai/tulu-v2-sft-mixture", split="train")

    # select only data where dataset is "lima"
    lima_data = data.filter(lambda x: x["dataset"] == "lima")

    # filter all messages that only contain a single question and answer pair
    # i.e., messages with exactly two entries (user and assistant)
    # this is to ensure that we only paraphrase single question answers
    single_question_answers = lima_data.filter(lambda x: len(x["messages"]) == 2)

    paraphrased = []

    for row in tqdm(single_question_answers, desc="Creating paraphrased dataset"):
        paraphrased.append(
            (
                row["id"],
                [
                    map_to_message_format("user", paraphrase_input(row["messages"][0]["content"])),
                    map_to_message_format("assistant", paraphrase_input(row["messages"][1]["content"]))
                ]
            )
        )

    lima_data_paraphrased = single_question_answers.add_column("paraphrased_id", [p[0] for p in paraphrased])
    lima_data_paraphrased = lima_data_paraphrased.add_column("paraphrased_messages", [p[1] for p in paraphrased])

    test = True
    for row in lima_data_paraphrased:
        test = test and (row["id"] == row["paraphrased_id"])

    print(f"All IDs match: {test}")

    lima_data_paraphrased = lima_data_paraphrased.remove_columns("dataset")
    lima_data_paraphrased = lima_data_paraphrased.remove_columns("paraphrased_id")

    lima_data_paraphrased = Dataset.from_list(list(lima_data_paraphrased))

    repo_id = "lukashinterleitner/LIMA-paraphrased-GPT-4o-mini"

    lima_data_paraphrased.push_to_hub(repo_id)
    print(f"Paraphrased dataset saved to: {repo_id}")


def create_model_generated_dataset():
    """Create model-generated dataset from paraphrased LIMA data."""

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load model and tokenizer
    model = get_model()
    tokenizer = get_tokenizer()

    # Load the paraphrased dataset
    try:
        paraphrased_dataset = get_paraphrased_dataset_tokenized(model, tokenizer, sample_size=None, partition_start=None, partition_end=None)
    except Exception as e:
        print(f"Error loading paraphrased dataset: {e}")
        print("Please create the paraphrased dataset first using --dataset-type paraphrased")
        return

    model_generated = []

    for row in tqdm(paraphrased_dataset, desc="Creating model-generated dataset"):
        try:
            generated_output = generate_model_output_from_paraphrased_sample(row, model, tokenizer)["model_generated_messages"]
            model_generated.append((row["id"], generated_output))
        except Exception as e:
            raise RuntimeError(f"Error generating output for sample {row['id']}: {e}") from e

    # Create a new dataset with model-generated content
    lima_data_model_generated = paraphrased_dataset.add_column("model_generated_id", [m[0] for m in model_generated])
    lima_data_model_generated = lima_data_model_generated.add_column("model_generated_messages", [m[1] for m in model_generated])

    # Verify IDs match
    test = True
    for row in lima_data_model_generated:
        test = test and (row["id"] == row["model_generated_id"])

    print(f"All IDs match: {test}")

    lima_data_model_generated = lima_data_model_generated.remove_columns("model_generated_id")

    lima_data_model_generated = Dataset.from_list(list(lima_data_model_generated))

    repo_id = get_model_generated_huggingface_dataset_path()

    lima_data_model_generated.push_to_hub(repo_id)
    print(f"Model-generated dataset saved to: {repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Generate paraphrased or model-generated datasets from LIMA data")
    parser.add_argument(
        "--dataset-type",
        choices=["paraphrased", "model-generated", "both"],
        required=True,
        help="Type of dataset to create: 'paraphrased' for paraphrased dataset, 'model-generated' for model-generated dataset, or 'both' for both datasets"
    )

    args = parser.parse_args()

    if args.dataset_type == "paraphrased":
        print("Creating paraphrased dataset...")
        create_paraphrased_dataset()
    elif args.dataset_type == "model-generated":
        print("Creating model-generated dataset...")
        create_model_generated_dataset()
    elif args.dataset_type == "both":
        print("Creating both paraphrased and model-generated datasets...")
        create_paraphrased_dataset()
        create_model_generated_dataset()


if __name__ == "__main__":
    main()
