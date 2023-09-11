#!/bin/bash
#SBATCH --job-name=swarm_class
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/donald.peltier/swarm/logs/swarm-ct%j.txt
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/4.10.3

source activate cs4321_peltier

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

python class_trans.py \
--model_dir="/home/donald.peltier/swarm/model/swarm_ctr_$(date +%Y-%m-%d_%H-%M-%S)/" \
--num_epochs=300 \
--batch_size=25 \
--data_path="/home/donald.peltier/swarm/data/data_6v6gsm_r300_s.npz" \
--window=-1