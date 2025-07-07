#!/bin/bash
# script name: paraphrase_both_datasets.sh
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-mounts=/srv/home/users/$USER/master-thesis:/app
#SBATCH --container-mount-home
#SBATCH --mem=124G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --output=./out/paraphrase_both_datasets/slurm-%j.out

echo "Starting creation of both paraphrased and model-generated datasets..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Create both datasets sequentially
python paraphrase.py --dataset-type both

echo "Both datasets creation completed at: $(date)"
