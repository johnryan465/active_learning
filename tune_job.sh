#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --job-name="active-learning-tunning"
#SBATCH --partition=msc
sh create_env.sh
source /scratch-ssd/oatml/miniconda3/bin/activate active_learning
python tune.py --data_path /scratch-ssd/oatml/data
