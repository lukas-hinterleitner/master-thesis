import argparse
from datasets import load_dataset
from src.paraphrasing import paraphrase_input
from src.model_operations import generate_model_output_from_paraphrased_sample, map_to_message_format
from src.storage import get_paraphrased_dataset_folder_path, get_model_generated_dataset_folder_path
from src.model import get_model, get_tokenizer

from tqdm import tqdm


def create_paraphrased_dataset():
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

    lima_data_paraphrased.save_to_disk(get_paraphrased_dataset_folder_path())
    print(f"Paraphrased dataset saved to: {get_paraphrased_dataset_folder_path()}")


def create_model_generated_dataset():
    """Create model-generated dataset from paraphrased LIMA data."""
    # Load the paraphrased dataset
    try:
        from datasets import load_from_disk
        paraphrased_dataset = load_from_disk(get_paraphrased_dataset_folder_path())
        print(f"Loaded paraphrased dataset from: {get_paraphrased_dataset_folder_path()}")
    except Exception as e:
        print(f"Error loading paraphrased dataset: {e}")
        print("Please create the paraphrased dataset first using --dataset-type paraphrased")
        return

    # Load model and tokenizer
    model = get_model()
    tokenizer = get_tokenizer()

    model_generated = []

    for row in tqdm(paraphrased_dataset, desc="Creating model-generated dataset"):
        try:
            generated_output = generate_model_output_from_paraphrased_sample(row, model, tokenizer)
            model_generated.append((row["id"], generated_output))
        except Exception as e:
            print(f"Error generating output for sample {row['id']}: {e}")
            # Use original paraphrased messages as fallback
            model_generated.append((row["id"], row["paraphrased_messages"]))

    # Create new dataset with model-generated content
    lima_data_model_generated = paraphrased_dataset.add_column("model_generated_id", [m[0] for m in model_generated])
    lima_data_model_generated = lima_data_model_generated.add_column("model_generated_messages", [m[1] for m in model_generated])

    # Verify IDs match
    test = True
    for row in lima_data_model_generated:
        test = test and (row["id"] == row["model_generated_id"])

    print(f"All IDs match: {test}")

    lima_data_model_generated = lima_data_model_generated.remove_columns("model_generated_id")

    lima_data_model_generated.save_to_disk(get_model_generated_dataset_folder_path())
    print(f"Model-generated dataset saved to: {get_model_generated_dataset_folder_path()}")


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
