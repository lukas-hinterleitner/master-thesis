#!/bin/bash
# script name: extract_gradients.sh
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-mount-home
#SBATCH --mem=124G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00
#SBATCH --container-workdir=$HOME/master-thesis
#SBATCH --nodelist=galadriel

python3 --version