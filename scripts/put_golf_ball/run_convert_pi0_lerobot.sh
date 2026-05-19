#!/bin/bash -l

#SBATCH --job-name=convert_pi0
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.out
#SBATCH --error=/scratch/grp/luo/shiyi/project/tele-gsy/script_results/%x_%j.err

set -e

PROJECT_ROOT=/scratch/grp/luo/shiyi/project/tele-gsy
OPENPI_ROOT=/scratch/grp/luo/shiyi/project/openpi
DATASET_NAME=put_golf_ball
OUTPUT_NAME=put_golf_ball_lerobot_no_tactile
REPO_ID=local/pi0_ur5e_put_golf_ball_no_tactile
DEFAULT_PROMPT="Put the golf ball into the small container on the tray"

INPUT_ROOT=${PROJECT_ROOT}/shared/data/bc_data/${DATASET_NAME}
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/${OUTPUT_NAME}
CONFIG_PATH=${PROJECT_ROOT}/learning/pi0_ur5e/configs/dataset_schema.yaml
CONVERT_SCRIPT=${PROJECT_ROOT}/learning/pi0_ur5e/scripts/convert_to_lerobot.py

cd "${OPENPI_ROOT}"
source /users/k25070928/miniconda3/etc/profile.d/conda.sh
conda activate tele

echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "Current directory: $(pwd)"
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "UV path: $(which uv)"
echo "Input root: ${INPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "================================"

echo "Checking GPU with nvidia-smi:"
nvidia-smi

echo "Converting raw trajectories to LeRobot/OpenPI format:"
uv run python "${CONVERT_SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --task-name "${DATASET_NAME}" \
  --repo-id "${REPO_ID}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --include-tactile false \
  --overwrite true

echo "Conversion finished."
