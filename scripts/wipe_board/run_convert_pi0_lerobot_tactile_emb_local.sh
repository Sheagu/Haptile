#!/bin/bash

set -e

PROJECT_ROOT=${PROJECT_ROOT:-/home/shiyigu/Documents/project/tele-gsy}
OPENPI_ROOT=${OPENPI_ROOT:-/home/shiyigu/Documents/project/openpi}
DATASET_NAME=fold_Tshirt
OUTPUT_NAME=fold_Tshirt_lerobot_tactile_emb
REPO_ID=local/pi0_ur5e_fold_Tshirt_tactile_emb
DEFAULT_PROMPT="Fold the t-shirt in half"
TACTILE_EMBEDDING_DIM=16

INPUT_ROOT=${PROJECT_ROOT}/shared/data/bc_data/${DATASET_NAME}
OUTPUT_ROOT=${PROJECT_ROOT}/outputs/${OUTPUT_NAME}
CONFIG_PATH=${PROJECT_ROOT}/learning/pi0_ur5e/configs/dataset_schema.yaml
CONVERT_SCRIPT=${PROJECT_ROOT}/learning/pi0_ur5e/scripts/convert_to_lerobot.py

cd "${OPENPI_ROOT}"

echo "================================"
echo "Local pi0 tactile-embedding conversion"
echo "Current directory: $(pwd)"
echo "Python path: $(command -v python || true)"
echo "UV path: $(command -v uv || true)"
echo "Input root: ${INPUT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Tactile embedding dim: ${TACTILE_EMBEDDING_DIM}"
echo "================================"

echo "Converting raw trajectories to LeRobot/OpenPI format with tactile embedding:"
uv run python "${CONVERT_SCRIPT}" \
  --input-root "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --config "${CONFIG_PATH}" \
  --task-name "${DATASET_NAME}" \
  --repo-id "${REPO_ID}" \
  --default-prompt "${DEFAULT_PROMPT}" \
  --action-mode joint_position_gripper \
  --include-tactile true \
  --tactile-feature-mode image_embedding \
  --tactile-embedding-dim "${TACTILE_EMBEDDING_DIM}" \
  --overwrite true

echo "Conversion finished."

echo "Checking converted dataset shapes:"
OUTPUT_ROOT="${OUTPUT_ROOT}" uv run python - <<'CHECK'
import json
import os
from pathlib import Path

root = Path(os.environ["OUTPUT_ROOT"])
info = json.loads((root / "meta" / "info.json").read_text())
print("state shape:", info["features"]["observation.state"]["shape"])
print("action shape:", info["features"]["action"]["shape"])
report_path = root / "conversion_report.json"
if report_path.exists():
    report = json.loads(report_path.read_text())
    print("tactile_shape:", report.get("tactile_shape"))
CHECK
