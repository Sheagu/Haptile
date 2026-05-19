#!/bin/bash -l

#SBATCH --job-name=prepare_cache
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

echo "Preparing cache data:"
python learning/dp/pipeline.py \
  --data_path /scratch/grp/luo/shiyi/project/tele-gsy/data_split/remove_sticky_note \
  --model_save_path /scratch/grp/luo/shiyi/project/tele-gsy/data/remove_sticky_note/ckpts/cache_prepare_dummy \
  --use_train_test_split True \
  --representation_type img-tactile_img-pos \
  --memmap_loader_path /scratch/grp/luo/shiyi/project/tele-gsy/data_split/remove_sticky_note_train/01-False-mem-img-tactile.dat \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --batch_size 32 \
  --num_workers 2 \
  --obs_horizon 1 \
  --pred_horizon 16 \
  --action_horizon 8 \
  --num_diffusion_iters 100 \
  --use_memmap_cache True \
  --load_img False \
  --gpu 0 \
  --prepare_cache_only True

echo "Job finished."
