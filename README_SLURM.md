# Distributed Processing with SLURM

This document explains how to run the gradient similarity computations and dataset preparation in a distributed manner using SLURM job scheduling.

## Overview

The code supports processing the entire dataset in parallel by dividing it into smaller partitions and distributing them across multiple SLURM jobs. This significantly speeds up computation compared to sequential execution, which is especially important for large language models and datasets.

## Setup

Ensure your environment is set up as described in the main `README.md`. Make sure all dependencies are installed and the .env file is properly configured.

## Dataset Preparation on SLURM

Before running the main analysis, you may need to prepare the datasets using SLURM. The repository includes pre-built SLURM scripts for dataset creation.

### Available Dataset Preparation Scripts

Located in `slurm_scripts/paraphrase_dataset/`:

1. **`slurm_paraphrase_dataset.sbatch`**
   - Creates paraphrased dataset from LIMA data
   - Resources: 64GB RAM, 8 CPUs, 12-hour time limit
   - No GPU required (uses OpenAI API)

2. **`slurm_model_generated_dataset.sbatch`**
   - Creates model-generated responses to paraphrased questions
   - Resources: 124GB RAM, 16 CPUs, 1 GPU, 24-hour time limit
   - **Requires paraphrased dataset to exist first**

3. **`slurm_paraphrase_both_datasets.sbatch`**
   - Creates both datasets sequentially
   - Resources: 124GB RAM, 16 CPUs, 1 GPU, 36-hour time limit
   - Handles complete dataset preparation pipeline

### Submitting Dataset Preparation Jobs

```bash
# Create only paraphrased dataset
sbatch slurm_scripts/paraphrase_dataset/slurm_paraphrase_dataset.sbatch

# Create only model-generated dataset (requires paraphrased dataset to exist)
sbatch slurm_scripts/paraphrase_dataset/slurm_model_generated_dataset.sbatch

# Create both datasets in sequence
sbatch slurm_scripts/paraphrase_dataset/slurm_paraphrase_both_datasets.sbatch
```

### Dataset Dependencies

**Important**: The model-generated dataset creation requires the paraphrased dataset to exist first. If you submit the model-generated job without the paraphrased dataset, it will fail with a clear error message.

Recommended workflow:
1. First run: `slurm_paraphrase_dataset.sbatch`
2. After completion, run: `slurm_model_generated_dataset.sbatch`
3. Or use: `slurm_paraphrase_both_datasets.sbatch` to handle both steps automatically

## Running Distributed Analysis Jobs

### Option 1: Submit all jobs at once

To generate and submit all jobs for all computation types:

```bash
bash slurm_scripts/batch_processing/compute_all.sh
```

This will generate and submit multiple SLURM jobs, each handling a partition of the data.

### Option 2: Submit specific job types

To generate and submit jobs for a specific computation type:

```bash
python slurm_scripts/batch_processing/compute.py --setting [paraphrased|model-generated] --computation-type [dot-product|gradient-similarity] [--use-random-projection] --num-partitions 10
```

Example:

```bash
python slurm_scripts/batch_processing/compute.py --setting paraphrased --computation-type gradient-similarity --num-partitions 10
```

### Option 3: Dry run (generate scripts without submitting)

To preview the SLURM scripts without submitting:

```bash
python slurm_scripts/batch_processing/compute.py --setting paraphrased --computation-type gradient-similarity --num-partitions 10 --dry-run
```

## Combining Results

After all jobs finish, combine partial outputs into final result files.

### Option 1: Combine all results at once

```bash
bash slurm_scripts/batch_processing/combine_all.sh
```

This invokes `combine.py` for each of the following result types:
- `paraphrased-gradient-similarities`
- `paraphrased-random-projected-gradient-similarities`
- `paraphrased-layer-dot-products`
- `model-generated-gradient-similarities`
- `model-generated-random-projected-gradient-similarities`
- `model-generated-layer-dot-products`

### Option 2: Combine a specific result type

```bash
python slurm_scripts/batch_processing/combine.py --result-type [result-type]
```

Valid `--result-type` values:
- `paraphrased-gradient-similarities`
- `paraphrased-random-projected-gradient-similarities`
- `paraphrased-layer-dot-products`
- `model-generated-gradient-similarities`
- `model-generated-random-projected-gradient-similarities`
- `model-generated-layer-dot-products`

Example:

```bash
python slurm_scripts/batch_processing/combine.py --result-type paraphrased-gradient-similarities
```

## Complete Workflow Example

Here's a complete workflow for running the entire analysis pipeline on SLURM:

```bash
# Step 1: Prepare datasets (if not already done)
sbatch slurm_scripts/paraphrase_dataset/slurm_paraphrase_both_datasets.sbatch

# Step 2: Wait for dataset preparation to complete, then run analysis
# Check job status: squeue -u $USER

# Step 3: Run distributed analysis
bash slurm_scripts/batch_processing/compute_all.sh

# Step 4: After all analysis jobs complete, combine results
bash slurm_scripts/batch_processing/combine_all.sh
```

## Checking Job Status

Use this to check SLURM job status:

```bash
squeue -u your_username
```

To view detailed information about a specific job:

```bash
scontrol show job [job_id]
```

## Output Files

### Dataset Preparation Logs
- `./out/paraphrase_dataset/slurm-[job_id].out`
- `./out/model_generated_dataset/slurm-[job_id].out`
- `./out/paraphrase_both_datasets/slurm-[job_id].out`

### Analysis Results
- Partial result files follow the naming pattern: `*_part_*_*.json`
- Combined results are saved with the suffix `_full.json`
- All results are stored in the appropriate subdirectories under the `data/` directory

## SLURM Script Details

### Dataset Preparation Scripts
Generated scripts include:
- Docker container configuration: `lukashinterleitner/master-thesis-data-science:latest`
- Resource allocation based on computational requirements
- Proper dependency handling between dataset types
- Comprehensive logging and error reporting

### Analysis Scripts
Generated scripts include:
- Docker container configuration
- Resource limits (CPU, GPU, memory, time)
- Output log paths
- Commands with partitioning parameters

## Troubleshooting

### Dataset Preparation Issues
- **"Paraphrased dataset not found"**: Ensure the paraphrased dataset job completed successfully before running model-generated dataset creation
- **OpenAI API errors**: Check that `OPENAI_API_KEY` is properly set in your `.env` file
- **Model loading errors**: Verify that `MT_MODEL_NAME` and `HF_TOKEN` are correctly configured

### Analysis Issues
- Check logs in the `./out/` directory
- Confirm dataset paths are correct
- Ensure output directories are writable
- If you encounter "Out of memory" errors, try reducing the partition size or requesting more memory
- For GPU-related issues, check if the correct GPU is being allocated using `nvidia-smi`

### Common Solutions
- **Missing datasets**: Run dataset preparation scripts first
- **Permission errors**: Check write permissions for output directories
- **Resource constraints**: Adjust memory/time limits in SLURM scripts as needed
- **Container issues**: Ensure Docker image is accessible and up-to-date
