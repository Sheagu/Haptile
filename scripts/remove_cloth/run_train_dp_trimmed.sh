#!/bin/bash -l

#SBATCH --job-name=train_dp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100_40g|h200"
#SBATCH --exclude=erc-hpc-comp031,erc-hpc-comp035
#SBATCH --cpus-per-task=24
#SBATCH --mem=24G
#SBATCH --time=06:00:00
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

echo "Training DP model:"
python learning/dp/pipeline.py \
  --data_path /scratch/grp/luo/shiyi/project/tele-gsy/data_split/remove_cloth_trimmed \
  --model_save_path /scratch/grp/luo/shiyi/project/tele-gsy/data/remove_cloth_trimmed/ckpts/dp_img_pos_tactile \
  --memmap_loader_path /scratch/grp/luo/shiyi/project/tele-gsy/data_split/remove_cloth_trimmed_train/01-False-mem-img-tactile.dat \
  --representation_type img-tactile_img-pos \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --touch_dim 60 \
  --obs_horizon 1 \
  --pred_horizon 16 \
  --action_horizon 8 \
  --batch_size 32 \
  --epochs 300 \
  --num_workers 16 \
  --use_train_test_split True \
  --use_memmap_cache True \
  --eval_freq 10 \
  --save_freq 10 \
  --num_diffusion_iters 100 \
  --load_img False \
  --use_wandb True \
  --wandb_entity_name shiyi_gu_seu \
  --wandb_project_name tele-gsy \
  --wandb_exp_name remove_cloth_dp_img_pos_tactile

echo "Job finished."
