#!/bin/bash -l

#SBATCH --job-name=split_data
#SBATCH --constraint="a100_40g|h200|a100_80g|l40s"
#SBATCH --exclude=erc-hpc-comp031,erc-hpc-comp035
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.err

set -e

cd /scratch/grp/luo/shiyi/project/tele-gsy
source /users/k25070928/miniconda3/etc/profile.d/conda.sh
conda activate tele

echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "================================"

echo "Checking GPU with nvidia-smi:"
nvidia-smi

echo "Spliting data into train/test sets:"
python workflow/split_data.py \
  --base_path /scratch/grp/luo/shiyi/project/tele-gsy/shared/data/bc_data\
  --output_path /scratch/grp/luo/shiyi/project/tele-gsy/data_split \
  --data_name fold_Tshirt

echo "Job finished."
