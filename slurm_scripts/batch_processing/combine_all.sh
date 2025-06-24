#!/bin/bash
# Script to combine all partial results after jobs complete

# Combine paraphrased gradient similarity results
echo "Combining paraphrased gradient similarity results..."
python slurm_scripts/batch_processing/combine.py --result-type paraphrased-gradient

# Combine paraphrased gradient similarity with random projection results
echo "Combining paraphrased gradient similarity with random projection results..."
python slurm_scripts/batch_processing/combine.py --result-type paraphrased-gradient-projection

# Combine paraphrased dot product results
echo "Combining paraphrased dot product results..."
python slurm_scripts/batch_processing/combine.py --result-type paraphrased-dot-product

# Combine model-generated gradient similarity results
echo "Combining model-generated gradient similarity results..."
python slurm_scripts/batch_processing/combine.py --result-type model-generated-gradient

# Combine model-generated gradient similarity with random projection results
echo "Combining model-generated gradient similarity with random projection results..."
python slurm_scripts/batch_processing/combine.py --result-type model-generated-gradient-projection

# Combine model-generated dot product results
echo "Combining model-generated dot product results..."
python slurm_scripts/batch_processing/combine.py --result-type model-generated-dot-product

echo "All results combined!"
