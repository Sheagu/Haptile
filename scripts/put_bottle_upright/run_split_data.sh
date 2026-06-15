#!/bin/bash -l

#SBATCH --job-name=split_data
#SBATCH --output=script_results/%x_%j.out
#SBATCH --error=script_results/%x_%j.err

set -e

cd /path/to/project
source /path/to/miniconda3/etc/profile.d/conda.sh
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
  --base_path /path/to/project/shared/data/bc_data\
  --output_path /path/to/project/data_split \
  --data_name put_bottle_upright

echo "Job finished."
