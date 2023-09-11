#!/bin/bash
#SBATCH --job-name=swarm_opt2
#SBATCH --array=1-200%300
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH -o swarm_opt2.out
#SBATCH --mem-per-cpu=4000

source /etc/profile
source ~/.bashrc

module use /share/modules/base

module load compile/gcc/7.2.0

module load app/matlab/R2017a

sed -n "${SLURM_ARRAY_TASK_ID}p" swarm_opt2.txt | /bin/bash