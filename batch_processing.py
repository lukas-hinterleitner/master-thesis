import argparse
import math
import os
import subprocess
from pathlib import Path

from datasets import load_from_disk

from src.config.storage import lima_paraphrased_dataset_path


def create_partitions(total_samples, num_partitions):
    """
    Create partition indices for distributing work across multiple jobs.

    Args:
        total_samples: Total number of samples in the dataset
        num_partitions: Number of partitions to create

    Returns:
        List of (start_idx, end_idx) tuples for each partition
    """
    partition_size = math.ceil(total_samples / num_partitions)
    partitions = []

    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = min((i + 1) * partition_size, total_samples)
        partitions.append((start_idx, end_idx))

    return partitions


def create_slurm_script(partition_idx, start_idx, end_idx, script_dir, setting, computation_type, use_random_projection):
    """
    Create a SLURM script for a specific partition.

    Args:
        partition_idx: Index of the current partition
        start_idx: Start index of the dataset partition
        end_idx: End index of the dataset partition
        script_dir: Directory to save the SLURM script
        setting: The computation setting (paraphrased or model-generated)
        computation_type: Type of computation (dot-product or gradient-similarity)
        use_random_projection: Whether to use random projection

    Returns:
        Path to the created SLURM script
    """
    os.makedirs(script_dir, exist_ok=True)

    # Create a descriptive name for the job
    job_name = f"{setting}_{computation_type}"
    if use_random_projection:
        job_name += "_random_projection"
    job_name += f"_part_{partition_idx}"

    script_path = os.path.join(script_dir, f"{job_name}.sbatch")

    # Prepare command line arguments
    cmd_args = f"--setting {setting} --computation-type {computation_type}"
    if use_random_projection:
        cmd_args += " --use-random-projection"
    cmd_args += f" --partition-start {start_idx} --partition-end {end_idx}"

    with open(script_path, 'w') as f:
        f.write(f"""#!/bin/bash
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-mounts=/srv/home/users/$USER/master-thesis:/app
#SBATCH --container-mount-home
#SBATCH --mem=86G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --output=./out/batch_processed/{job_name}/slurm-%j.out
#SBATCH --job-name={job_name}

python3 main.py {cmd_args}
""")

    return script_path


def submit_slurm_jobs(slurm_scripts):
    """
    Submit all SLURM scripts as jobs.

    Args:
        slurm_scripts: List of paths to SLURM scripts

    Returns:
        List of job IDs
    """
    job_ids = []

    for script in slurm_scripts:
        result = subprocess.run(['sbatch', script], capture_output=True, text=True)
        if result.returncode == 0:
            # Extract job ID from sbatch output (format: "Submitted batch job 12345")
            job_id = result.stdout.strip().split()[-1]
            job_ids.append(job_id)
            print(f"Submitted job {job_id} with script {script}")
        else:
            print(f"Error submitting job with script {script}: {result.stderr}")

    return job_ids


def main():
    parser = argparse.ArgumentParser(description="Create and submit SLURM batch jobs for distributed processing")

    parser.add_argument(
        "--setting",
        type=str,
        choices=["paraphrased", "model-generated"],
        required=True,
        help="Specify the setting for the computation: model-generated or paraphrased."
    )

    parser.add_argument(
        "--computation-type",
        type=str,
        choices=["dot-product", "gradient-similarity"],
        required=True,
        help="Specify the computation type: dot-product or gradient-similarity."
    )

    parser.add_argument(
        "--use-random-projection",
        action="store_true",
        help="Use random projection for the computation. Only relevant for gradient similarity."
    )

    parser.add_argument(
        "--num-partitions",
        type=int,
        default=10,
        help="Number of partitions to create."
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create SLURM scripts but don't submit jobs."
    )

    args = parser.parse_args()

    # Load dataset to get total number of samples
    dataset = load_from_disk(lima_paraphrased_dataset_path)
    total_samples = len(dataset)
    print(f"Total samples in dataset: {total_samples}")

    # Create partitions
    partitions = create_partitions(total_samples, args.num_partitions)
    print(f"Created {len(partitions)} partitions")

    # Create directory for SLURM scripts
    script_dir = Path("slurm_scripts")
    script_dir.mkdir(exist_ok=True)

    # Create SLURM scripts for each partition
    slurm_scripts = []
    for i, (start_idx, end_idx) in enumerate(partitions):
        script_path = create_slurm_script(
            i, start_idx, end_idx, script_dir, 
            args.setting, args.computation_type, args.use_random_projection
        )
        slurm_scripts.append(script_path)
        print(f"Created SLURM script for partition {i}: {start_idx}-{end_idx}")

    # Submit jobs if not a dry run
    if not args.dry_run:
        job_ids = submit_slurm_jobs(slurm_scripts)
        print(f"Submitted {len(job_ids)} jobs")
    else:
        print("Dry run - SLURM scripts created but not submitted")


if __name__ == "__main__":
    main()
