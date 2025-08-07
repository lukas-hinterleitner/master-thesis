#!/bin/bash
# script name: random_projection_full_gradient.sh
#SBATCH --container-image=lukashinterleitner/master-thesis-data-science:latest
#SBATCH --container-mounts=/srv/home/users/$USER/master-thesis:/app
#SBATCH --container-mount-home
#SBATCH --mem=124G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=0-24:00:00
#SBATCH --nodelist=dgx-h100-em2
#SBATCH --output=./out/random_projection_full_gradient/slurm-%j.out


# Execute the script
python compute_random_projection.py --model_name amd/AMD-OLMo-1B-SFT

