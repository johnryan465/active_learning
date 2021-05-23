#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:8
#SBATCH --job-name="active-learning-tuning"
#SBATCH --partition=htc
module load python/anaconda3/2020.11
module load gpu/cuda/10.1.243
module load gpu/cudnn/7.6.5__cuda-10.1

source activate $DATA/conda_envs/active_learning

python tune.py --data_path $DATA/data
