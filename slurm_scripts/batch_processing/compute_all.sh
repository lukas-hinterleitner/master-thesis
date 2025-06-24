#!/bin/bash
# Script to generate and submit all SLURM jobs for distributed processing

# Set number of partitions (adjust as needed based on dataset size and available resources)
NUM_PARTITIONS=50

# Generate and submit jobs for paraphrased gradient similarity
echo "Generating paraphrased gradient similarity jobs..."
python slurm_scripts/batch_processing/compute.py --setting paraphrased --computation-type gradient-similarity --num-partitions $NUM_PARTITIONS

# Generate and submit jobs for paraphrased gradient similarity with random projection
echo "Generating paraphrased gradient similarity with random projection jobs..."
python slurm_scripts/batch_processing/compute.py --setting paraphrased --computation-type gradient-similarity --use-random-projection --num-partitions $NUM_PARTITIONS

# Generate and submit jobs for paraphrased dot product
echo "Generating paraphrased dot product jobs..."
python slurm_scripts/batch_processing/compute.py --setting paraphrased --computation-type dot-product --num-partitions $NUM_PARTITIONS

# Generate and submit jobs for model-generated gradient similarity
echo "Generating model-generated gradient similarity jobs..."
python slurm_scripts/batch_processing/compute.py --setting model-generated --computation-type gradient-similarity --num-partitions $NUM_PARTITIONS

# Generate and submit jobs for model-generated gradient similarity with random projection
echo "Generating model-generated gradient similarity with random projection jobs..."
python slurm_scripts/batch_processing/compute.py --setting model-generated --computation-type gradient-similarity --use-random-projection --num-partitions $NUM_PARTITIONS

# Generate and submit jobs for model-generated dot product
echo "Generating model-generated dot product jobs..."
python slurm_scripts/batch_processing/compute.py --setting model-generated --computation-type dot-product --num-partitions $NUM_PARTITIONS

echo "All jobs submitted!"
