#!/bin/bash
#SBATCH --job-name=gen_saq
#SBATCH --output=jupyter_logs/log-%J.txt
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=6:00:00

source /data/phillip_guo/miniconda3/etc/profile.d/conda.sh
conda activate hp-unlrn
python -m tasks.hp.HPSAQ