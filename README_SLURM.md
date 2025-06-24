# Distributed Processing with SLURM

This document explains how to run the computations in a distributed manner using SLURM job scheduling.

## Overview

The code has been updated to support processing the entire dataset in parallel by dividing it into smaller partitions and distributing them across multiple SLURM jobs. This approach significantly speeds up the computation compared to processing the entire dataset sequentially.

## Setup

Ensure you have set up the environment as described in the main README.md file.

## Running Distributed Jobs

### Option 1: Submit all jobs at once

To generate and submit all jobs for all computation types:

```bash
bash slurm_scripts/run_all.sh
```

This script will create and submit multiple SLURM jobs, dividing the dataset into partitions.

### Option 2: Submit specific job types

To generate and submit jobs for a specific computation type:

```bash
python batch_processing.py --setting [paraphrased|model-generated] --computation-type [dot-product|gradient-similarity] [--use-random-projection] --num-partitions 10
```

For example, to submit jobs for paraphrased gradient similarity with 10 partitions:

```bash
python batch_processing.py --setting paraphrased --computation-type gradient-similarity --num-partitions 10
```

### Option 3: Dry run (generate scripts without submitting)

To generate SLURM scripts without submitting them (useful for review):

```bash
python batch_processing.py --setting paraphrased --computation-type gradient-similarity --num-partitions 10 --dry-run
```

## Combining Results

After all jobs have completed, you need to combine the partial results into a single result file.

### Option 1: Combine all results at once

```bash
bash slurm_scripts/combine_all_results.sh
```

This will combine all partial results for all computation types.

### Option 2: Combine specific result types

```bash
python combine_results.py --result-type [result-type]
```

Where [result-type] is one of:
- paraphrased-gradient
- paraphrased-gradient-projection
- paraphrased-dot-product
- model-generated-gradient
- model-generated-gradient-projection
- model-generated-dot-product

For example:

```bash
python combine_results.py --result-type paraphrased-gradient
```

## Checking Job Status

To check the status of your submitted jobs:

```bash
squeue -u your_username
```

## Output Files

Partial results will be stored in the same directory structure as the original code, but with partition-specific filenames. Combined results will have "_full" in the filename instead of the sample size.

## SLURM Script Structure

The generated SLURM scripts include:

- Container settings for Docker
- Memory, CPU, and GPU resource allocations
- Time limits
- Output file paths
- Command to run the code with specific partition parameters

## Troubleshooting

- Check SLURM output files in the `./out/` directory for error messages
- Verify that the dataset path is correct and accessible
- Ensure the code has write permissions to the output directories
