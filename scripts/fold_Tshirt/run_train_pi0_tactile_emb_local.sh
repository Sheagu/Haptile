#!/bin/bash

set -e

PROJECT_ROOT=${PROJECT_ROOT:-/home/shiyigu/Documents/project/tele-gsy}
OPENPI_ROOT=${OPENPI_ROOT:-/home/shiyigu/Documents/project/openpi}
DATASET_NAME=fold_Tshirt
LEROBOT_REPO_ID=local/pi0_ur5e_fold_Tshirt_tactile_emb
EXP_NAME=fold_Tshirt_pi0_base_tactile_emb_lora
DEFAULT_PROMPT="Fold the t-shirt in half"
DRY_RUN=${DRY_RUN:-true}
WANDB=${WANDB:-false}

DATASET_ROOT=${PROJECT_ROOT}/outputs/${DATASET_NAME}_lerobot_tactile_emb
OUTPUT_DIR=${PROJECT_ROOT}/outputs/pi0_${DATASET_NAME}_tactile_emb_lora

cd "${PROJECT_ROOT}"

echo "================================"
echo "Local pi0 tactile-embedding training"
echo "Current directory: $(pwd)"
echo "Python path: $(command -v python || true)"
echo "UV path: $(command -v uv || true)"
echo "Dataset root: ${DATASET_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Dry run: ${DRY_RUN}"
echo "WandB enabled: ${WANDB}"
echo "================================"

echo "Launching pi0 LoRA training helper with tactile embedding:"
bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --openpi-root "${OPENPI_ROOT}" \
  --repo-id "${LEROBOT_REPO_ID}" \
  --exp-name "${EXP_NAME}" \
  --steps 30000 \
  --batch-size 16 \
  --model-family pi0 \
  --pi05 false \
  --lora true \
  --camera-padding-strategy zeros \
  --use-delta-actions true \
  --include-tactile true \
  --wandb "${WANDB}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --dry-run "${DRY_RUN}"

echo "Job finished."
