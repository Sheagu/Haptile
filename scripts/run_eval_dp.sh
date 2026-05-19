#!/bin/bash -l

#SBATCH --job-name=eval_dp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40g|h200"
#SBATCH --exclude=erc-hpc-comp031,erc-hpc-comp035
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=01:00:00
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

echo "Testing DP model:"
python eval_dir.py \
  --ckpt_path /scratch/grp/luo/shiyi/project/tele-gsy/data/rubiks_cube_trimmed/ckpts/dp_img_pos_delta_fast/0428_205543_RIzG-camera=01-identity=False-repr=IP-oh=2-ah=4-ph=12-prefix=None-do=0.0-imgos=64-wd=1e-05-use_ddim=False-binarize_touch=False-posdelta/last.ckpt \
  --eval_dir /scratch/grp/luo/shiyi/project/tele-gsy/data_split/rubiks_cube_trimmed_test \
  --save_path /scratch/grp/luo/shiyi/project/tele-gsy/data/rubiks_cube_trimmed/eval_results/eval_last.pkl

echo "Job finished."
