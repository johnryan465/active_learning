#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name="active-learning"
#SBATCH --partition=msc

export CONDA_ENVS_PATH=/scratch/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs

/scratch-ssd/oatml/run_locked.sh /scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate active_learning
pip install batchbald_redux==2.0.5 --no-deps
pip install gpytorch==1.4.1
python main.py --name log_image --description "BatchBALD vDUQ" --initial_per_class 2 --smoke_test True --aquisition_size 4 batchbald --batch_size 64 --epochs 1 vduq --lr 0.001 --coeff 9
