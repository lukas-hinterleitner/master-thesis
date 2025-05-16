#!/bin/bash
# script name: gradient_similarities_paraphrased.sh
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-workdir=$HOME/master-thesis
#SBATCH --container-mount-home
#SBATCH --mem=124G
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:05:00
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --output=./out/slurm-%j.out

python3 --version
ls -a
cd ..
ls -a
cd ..
ls -a
# python3 application/gradient_similarities_paraphrased.py