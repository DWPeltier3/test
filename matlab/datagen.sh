#!/bin/bash
#SBATCH --job-name=swarm_savedata
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --partition=beards

source /etc/profile
source ~/.bashrc

module use /share/modules/base

module load compile/gcc/7.2.0

module load app/matlab/R2017a

matlab -nodisplay -nojvm -nosplash -r "datagen"