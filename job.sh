#!/bin/bash#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name="active-learning"
export CONDA_ENVS_PATH=/scratch/$USER/conda_envs
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs/scratch-ssd/oatml/scripts/run_locked.sh
/scratch-ssd/oatml/miniconda3/bin/conda-env update -f environment.yml
source /scratch-ssd/oatml/miniconda3/bin/activate active_learning
srun python main.py --name "vduq_bb_0.001" --description "BatchBALD vDUQ" --initial_per_class 2  --aquisition_size 5 batchbald  --batch_size 64 --epochs 100 vduq --lr 0.001
