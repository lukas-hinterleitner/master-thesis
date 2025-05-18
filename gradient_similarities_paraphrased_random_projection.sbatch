#!/bin/bash
# script name: gradient_similarities_paraphrased_random_projection.sh
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-mounts=/srv/home/users/$USER/master-thesis:/app
#SBATCH --container-mount-home
#SBATCH --mem=124G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --output=./out/slurm-%j.out

#python3 --version

python3 application/gradient_similarities_paraphrased_random_projection.py