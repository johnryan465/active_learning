#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --job-name="active-learning-tunning"
#SBATCH --partition=msc
export CONDA_ENVS_PATH=/scratch/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs/scratch-ssd/oatml/scripts/run_locked.sh
/scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate active_learning
pip install batchbald_redux==2.0.5 --no-deps
python tune.py --data_path /scratch-ssd/oatml/data
