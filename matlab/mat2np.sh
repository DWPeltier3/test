#!/bin/bash
#SBATCH --job-name=swarm_mat2np
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/donald.peltier/swarm/logs/mat2np%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/23.1.0

source activate swarm

python mat2np4.py --scaling_factor 0