#!/bin/bash
#SBATCH --job-name=swarm-class
#SBATCH --output=/home/donald.peltier/swarm/logs/swarm-class_CNNnoise%a.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=beards
#SBATCH --array=1-50

. /etc/profile

module load lang/miniconda3/23.1.0
module load app/graphviz/8.0.5

source activate swarm

python class.py \
--mode="train" \
--trained_model="/home/donald.peltier/swarm/model/swarm_class11-01_11-08-31_RT_TRVmhWF_rs/model.keras" \
--model_dir="/home/donald.peltier/swarm/model/swarm_class_CNNnoise${SLURM_ARRAY_TASK_ID}/" \
--data_path="/home/donald.peltier/swarm/data/data_10v10_r4800s_4cl_a10_noise${SLURM_ARRAY_TASK_ID}.npz" \
--window=-1 \
--model_type="cn" \
--output_type="mh" \
--output_length="vec" \
--dropout=0. \
--kernel_initializer="he_normal" \
--kernel_regularizer="none" \
--optimizer="adam" \
--initial_learning_rate=0.0001 \
--callback_list="checkpoint, early_stopping, csv_log" \
--patience=50 \
--tune_type="r" \
--tune_epochs=1000 \
--num_epochs=1000 \
--batch_size=50 \
--train_val_split=0.2 \

## NOTES
# mode = 'train' or 'predict'
# window = -1 uses full window
# model_type = 'fc'=fully connect, 'cn'=CNN, 'fcn'=FCN, 'res'=ResNet, 'lstm'=long short term memory, 'tr'=transformer
# output_type = 'mc'=multiclass, 'ml'=multilabel, 'mh'=multihead
# output_length = 'vec'=vector (final only), 'seq'=sequence (every time step) **
#                                              ** only "lstm" or "tr" can have "seq" output
# kernel_initializer = "glorot_normal" "he_uniform/normal"
# kernel_regularizer = "none" "l1" "l2" "l1_l2"
#
# callback_list="checkpoint, early_stopping, csv_log"
#
# tune_type = tuner type: "r"=random, "b"=bayesian, "h"=hyperband