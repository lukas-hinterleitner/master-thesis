#!/bin/bash
# script name: model_generated_dataset.sh
#SBATCH --job-name=model_generated_dataset
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-mounts=/srv/home/users/$USER/master-thesis:/app
#SBATCH --container-mount-home
#SBATCH --mem=124G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --nodelist=dgx1
#SBATCH --output=./out/model_generated_dataset/slurm-%j.out

echo "Starting model-generated dataset creation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Create model-generated dataset (requires paraphrased dataset to exist)
python paraphrase.py --dataset-type model-generated

echo "Model-generated dataset creation completed at: $(date)"
