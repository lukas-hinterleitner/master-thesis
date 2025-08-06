import argparse
import re
from pathlib import Path


def get_execution_time(file_path):
    """
    Extracts execution time from a SLURM output file.

    Args:
        file_path (str): The path to the SLURM output file.

    Returns:
        float: The execution time in seconds, or 0.0 if not found.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            match = re.search(r"Elapsed time: ([\d\.]+)", content)
            if match:
                return float(match.group(1))
    except (IOError, ValueError) as e:
        print(f"Could not read or parse {file_path}: {e}")
    return 0.0


def main():
    """
    Main function to calculate the total execution time from SLURM logs.
    """
    parser = argparse.ArgumentParser(
        description="Calculate total execution time from SLURM log files."
    )
    parser.add_argument(
        "--setting",
        type=str,
        required=True,
        choices=["paraphrased", "model-generated"],
        help="The computation setting (e.g., 'paraphrased', 'model-generated')."
    )
    parser.add_argument(
        "--computation-type",
        type=str,
        required=True,
        choices=["dot-product", "gradient-similarity"],
        help="The type of computation (e.g., 'dot-product', 'gradient-similarity')."
    )
    parser.add_argument(
        "--use-random-projection",
        action="store_true",
        help="Flag to indicate if random projection was used."
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        required=True,
        help="The number of partitions used in the computation."
    )
    args = parser.parse_args()

    folder_name = f"{args.setting}_{args.computation_type}"
    if args.use_random_projection:
        folder_name += "_random_projection"

    log_dir = Path("out/batch_processed") / folder_name
    if not log_dir.is_dir():
        print(f"Error: Log directory not found at '{log_dir}'")
        return

    total_execution_time = 0.0
    processed_partitions = set()
    job_ids = set()

    for partition_folder in sorted(log_dir.iterdir()):
        if partition_folder.is_dir() and partition_folder.name.startswith("part_"):
            try:
                partition_idx = int(partition_folder.name.split('_')[1])
                
                log_files = list(partition_folder.glob("slurm-*.out"))
                if not log_files:
                    print(f"Warning: No log files found in {partition_folder}")
                    continue

                # Find the most recent log file based on job ID
                latest_log_file = max(log_files, key=lambda p: int(re.search(r'slurm-(\d+)\.out', str(p)).group(1)))
                job_id = int(re.search(r'slurm-(\d+)\.out', str(latest_log_file)).group(1))
                
                if job_id in job_ids:
                    print(f"Warning: Duplicate job ID {job_id} found. Skipping file {latest_log_file}.")
                    continue

                execution_time = get_execution_time(latest_log_file)
                if execution_time > 0:
                    total_execution_time += execution_time
                    processed_partitions.add(partition_idx)
                    job_ids.add(job_id)
                else:
                    print(f"Warning: Could not find execution time in {latest_log_file}")


            except (ValueError, IndexError):
                print(f"Warning: Could not parse partition index from folder name '{partition_folder.name}'")

    if len(processed_partitions) != args.num_partitions:
        print(f"Warning: Found logs for {len(processed_partitions)} partitions, but expected {args.num_partitions}.")
        missing = set(range(args.num_partitions)) - processed_partitions
        if missing:
            print(f"Missing partitions: {sorted(list(missing))}")

    # Check for contiguous job IDs
    if job_ids:
        min_job_id = min(job_ids)
        max_job_id = max(job_ids)
        expected_job_ids = set(range(min_job_id, max_job_id + 1))

        if job_ids != expected_job_ids:
            print("\nWarning: Job IDs are not contiguous.")
            missing_job_ids = sorted(list(expected_job_ids - job_ids))
            if missing_job_ids:
                print(f"Missing job IDs in sequence: {missing_job_ids}")

    
    print(f"\nProcessed {len(job_ids)} unique job log(s).")
    print(f"Total execution time for '{folder_name}': {total_execution_time:.2f} seconds")
    print(f"This is equal to {total_execution_time / 3600:.2f} hours")


if __name__ == "__main__":
    main()
