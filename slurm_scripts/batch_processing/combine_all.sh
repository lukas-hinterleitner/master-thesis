#!/bin/bash
# Script to combine all partial results after jobs complete

# Combine paraphrased gradient similarity results
echo "Combining paraphrased gradient similarity results..."
python slurm_scripts/batch_processing/combine.py --result-type paraphrased-gradient-similarities

# Combine paraphrased gradient similarity with random projection results
echo "Combining paraphrased gradient similarity with random projection results..."
python slurm_scripts/batch_processing/combine.py --result-type paraphrased-random-projected-gradient-similarities

# Combine paraphrased dot product results
echo "Combining paraphrased dot product results..."
python slurm_scripts/batch_processing/combine.py --result-type paraphrased-layer-dot-products

# Combine model-generated gradient similarity results
echo "Combining model-generated gradient similarity results..."
python slurm_scripts/batch_processing/combine.py --result-type model-generated-gradient-similarities

# Combine model-generated gradient similarity with random projection results
echo "Combining model-generated gradient similarity with random projection results..."
python slurm_scripts/batch_processing/combine.py --result-type model-generated-random-projected-gradient-similarities

# Combine model-generated dot product results
echo "Combining model-generated dot product results..."
python slurm_scripts/batch_processing/combine.py --result-type model-generated-layer-dot-products

echo "All results combined!"