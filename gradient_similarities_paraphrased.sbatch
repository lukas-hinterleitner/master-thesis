#!/bin/bash
# script name: gradient_similarities_paraphrased.sh
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-mounts=/srv/home/users/$USER/master-thesis:/app
#SBATCH --container-mount-home
#SBATCH --mem=124G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --output=./out/gradient_similarities_paraphrased/slurm-%j.out

python3 main.py --setting paraphrased --computation-type gradient-similarity