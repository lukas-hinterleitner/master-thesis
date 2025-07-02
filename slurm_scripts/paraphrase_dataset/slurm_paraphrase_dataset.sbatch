#!/bin/bash
# script name: paraphrase_dataset.sh
#SBATCH --job-name=paraphrase_dataset
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-mounts=/srv/home/users/$USER/master-thesis:/app
#SBATCH --container-mount-home
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=dgx-h100-em2,dgx1
#SBATCH --output=./out/paraphrase_dataset/slurm-%j.out


echo "Starting paraphrased dataset creation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Create paraphrased dataset
python paraphrase.py --dataset-type paraphrased

echo "Paraphrased dataset creation completed at: $(date)"
