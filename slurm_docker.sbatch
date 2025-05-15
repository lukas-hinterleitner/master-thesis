#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-mount-home
#SBATCH --mem=124G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-4:00:00
#SBATCH --container-workdir=$HOME/master-thesis
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --output=./out/slurm-%j.out

python3 --version