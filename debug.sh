#!/bin/bash
#SBATCH --job-name=swarm_class
#SBATCH --output=../logs/debug/swarm-cfc-debug%j.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=beards

. /etc/profile

module load lang/miniconda3/23.1.0

source activate swarm

python -m debugpy --wait-for-client --listen 0.0.0.0:54321 \
class.py \
--mode="train" \
--trained_model= \
--model_dir="/home/donald.peltier/swarm/model/swarm_cnn_$(date +%Y-%m-%d_%H-%M-%S)/" \
--data_path="/home/donald.peltier/swarm/data/data_10v10_r4800s_4cl_a10.npz" \
--window=20 \
--model_type="cn" \
--output_type="mc" \
--dropout=0.2 \
--optimizer="nadam" \
--initial_learning_rate=0.001 \
--callback_list="checkpoint, early_stopping, csv_log" \
--patience=50 \
--num_epochs=1000 \
--batch_size=50 \
--train_val_split=0.2 \

## NOTES
# window = -1 uses full window
# model_type = 'fc'=fully connect, 'cn'=CNN, 'tr'=transformer
# output_type = 'mc'=multiclass, 'ml'=multilabel, 'mo'=multiout, 'ser'=series