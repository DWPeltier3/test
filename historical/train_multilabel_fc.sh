#!/bin/bash
#SBATCH --job-name=swarm_class
#SBATCH --output=/home/donald.peltier/swarm/logs/swarm-cML%j.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/23.1.0

source activate swarm

python multilabel_fc.py \
--model_dir="/home/donald.peltier/swarm/model/swarm_cfc_$(date +%Y-%m-%d_%H-%M-%S)/" \
--data_path="/home/donald.peltier/swarm/data/data_10v10_r4800s_4cl_a10.npz" \
--dropout=0.2 \
--num_epochs=1000 \
--batch_size=25 \
--window=40 # -1 uses full window