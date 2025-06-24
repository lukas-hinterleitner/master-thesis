# Distributed Processing with SLURM

This document explains how to run the computations in a distributed manner using SLURM job scheduling.

## Overview

The code supports processing the entire dataset in parallel by dividing it into smaller partitions and distributing them across multiple SLURM jobs. This significantly speeds up computation compared to sequential execution.

## Setup

Ensure your environment is set up as described in the main `README.md`.

## Running Distributed Jobs

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

## Checking Job Status

Use this to check SLURM job status:

```bash
squeue -u your_username
```

## Output Files

- Partial result files follow the naming pattern: `*_part_*_*.json`.
- Combined results are saved with the suffix `_full.json`.

## SLURM Script Details

Generated scripts include:
- Docker container configuration
- Resource limits (CPU, GPU, memory, time)
- Output log paths
- Commands with partitioning parameters

## Troubleshooting

- Check logs in the `./out/` directory
- Confirm dataset paths are correct
- Ensure output directories are writable