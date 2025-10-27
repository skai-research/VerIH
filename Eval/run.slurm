#!/bin/bash
#SBATCH --account=PAS2836
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:2
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --output=./%j.out
#SBATCH --partition=quad

source ~/.bashrc
conda activate verih_eval
bash ~/sys_check/run.sh
