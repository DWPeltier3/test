#!/bin/bash
#SBATCH --job-name=swarm-class
#SBATCH --output=/home/donald.peltier/swarm/logs/swarm-class_%a.txt
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=beards
#SBATCH --array=1-24

. /etc/profile

module load lang/miniconda3/23.1.0
module load app/graphviz/8.0.5

source activate swarm

# Number of array = #model type * #window * #swarm size
# Define model types and output lengths
model_types=("cn" "fcn" "tr" "tr")
output_lengths=("vec" "vec" "vec" "seq")
# Define windows and swarm sizes
windows=(20 -1)
swarm_sizes=(25 50 75)

# Calculate the combination index
array_index=$((SLURM_ARRAY_TASK_ID - 1))

# Determine the combination
model_output_index=$((array_index / 6)) # 6 = 2 windows * 3 swarm sizes
window_index=$(((array_index / 3) % 2)) # Cycles every 3 jobs, toggles every 6
swarm_index=$((array_index % 3))       # Cycles every job, repeats every 3

# Get model type, output length, and other parameters from the calculated index
model_type=${model_types[$model_output_index]}
output_length=${output_lengths[$model_output_index]}
window=${windows[$window_index]}
swarm_size=${swarm_sizes[$swarm_index]}

python class.py \
--mode="train" \
--trained_model="/home/donald.peltier/swarm/model/swarm_class11-01_11-08-31_RT_TRVmhWF_rs/model.keras" \
--model_dir="/home/donald.peltier/swarm/model/swarm_class$(date +%m-%d_%H-%M-%S)_${model_type}_${output_length}_w${window}_${swarm_size}/" \
--data_path="/home/donald.peltier/swarm/data/data_${swarm_size}v${swarm_size}_r4800s_4cl_a10.npz" \
--features="v" \
--output_type="mh" \
--dropout=0. \
--kernel_initializer="he_normal" \
--kernel_regularizer="none" \
--optimizer="adam" \
--initial_learning_rate=0.0001 \
--callback_list="checkpoint, early_stopping, csv_log" \
--patience=50 \
--num_epochs=1000 \
--batch_size=50 \
--val_split=0.2 \
--tune_type="r" \
--tune_epochs=1000 \
--model_type="${model_type}" \
--output_length="${output_length}" \
--window=${window} \
