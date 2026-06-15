#!/bin/bash -l

#SBATCH --job-name=prepare_cache
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

echo "Preparing cache data:"
python learning/dp/pipeline.py \
  --data_path /path/to/project/data_split/wipe_board \
  --model_save_path /path/to/project/data/wipe_board/ckpts/cache_prepare_dummy \
  --use_train_test_split True \
  --representation_type img-pos \
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
