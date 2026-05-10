#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET_ROOT=""
OUTPUT_DIR=""
OPENPI_ROOT=""
CONFIG_NAME="pi0_ur5e_cup"
EXP_NAME="ur5e_cup_pi05_base"
STEPS="30000"
BATCH_SIZE="16"
ACTION_HORIZON="50"
MAX_TOKEN_LEN=""
KEEP_PERIOD="1000"
LORA="true"
PI05="true"
MODEL_FAMILY=""
USE_DELTA_ACTIONS="true"
FREEZE_MODE="default"
XLA_PREALLOCATE="unset"
WANDB="false"
RESUME="false"
OVERWRITE="true"
INCLUDE_TACTILE="false"
ACTION_FORMAT="joint_position_gripper"
CAMERA_PADDING="zeros"
DEFAULT_PROMPT="pick up the paper cup and place it on the target"
LEROBOT_REPO_ID="local/pi0_ur5e_cup"
DRY_RUN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset-root) DATASET_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --openpi-root) OPENPI_ROOT="$2"; shift 2 ;;
    --openpi-config|--config-name) CONFIG_NAME="$2"; shift 2 ;;
    --exp-name) EXP_NAME="$2"; shift 2 ;;
    --steps) STEPS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --action-horizon) ACTION_HORIZON="$2"; shift 2 ;;
    --max-token-len) MAX_TOKEN_LEN="$2"; shift 2 ;;
    --keep-period) KEEP_PERIOD="$2"; shift 2 ;;
    --lora) LORA="$2"; shift 2 ;;
    --pi05) PI05="$2"; shift 2 ;;
    --model-family) MODEL_FAMILY="$2"; shift 2 ;;
    --use-delta-actions) USE_DELTA_ACTIONS="$2"; shift 2 ;;
    --freeze-mode) FREEZE_MODE="$2"; shift 2 ;;
    --xla-preallocate) XLA_PREALLOCATE="$2"; shift 2 ;;
    --wandb) WANDB="$2"; shift 2 ;;
    --resume) RESUME="$2"; shift 2 ;;
    --overwrite) OVERWRITE="$2"; shift 2 ;;
    --include-tactile) INCLUDE_TACTILE="$2"; shift 2 ;;
    --action-format) ACTION_FORMAT="$2"; shift 2 ;;
    --camera-padding-strategy) CAMERA_PADDING="$2"; shift 2 ;;
    --default-prompt) DEFAULT_PROMPT="$2"; shift 2 ;;
    --lerobot-repo-id|--repo-id) LEROBOT_REPO_ID="$2"; shift 2 ;;
    --dry-run) DRY_RUN="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$DATASET_ROOT" || -z "$OUTPUT_DIR" || -z "$OPENPI_ROOT" ]]; then
  echo "Required: --dataset-root --output-dir --openpi-root" >&2
  exit 2
fi

UV_BIN="${UV_BIN:-}"
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "$UV_BIN" && -x "$HOME/.local/bin/uv" ]]; then
  UV_BIN="$HOME/.local/bin/uv"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "Could not find uv. Install uv or set UV_BIN=/path/to/uv." >&2
  exit 2
fi

DATASET_ROOT="$(realpath "$DATASET_ROOT")"
OUTPUT_DIR="$(mkdir -p "$OUTPUT_DIR" && realpath "$OUTPUT_DIR")"
OPENPI_ROOT="$(realpath "$OPENPI_ROOT")"
STATE_DIM="$("$OPENPI_ROOT/.venv/bin/python" - "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

info_path = Path(sys.argv[1]) / "meta" / "info.json"
info = json.loads(info_path.read_text())
shape = info["features"]["observation.state"]["shape"]
print(shape[0])
PY
)"

mkdir -p "$OUTPUT_DIR/logs"
cp -f "$DATASET_ROOT/conversion_report.json" "$OUTPUT_DIR/conversion_report.json" 2>/dev/null || true
if [[ -f "$DATASET_ROOT/conversion_report.json" && "$ACTION_FORMAT" == "auto" ]]; then
  ACTION_FORMAT="$("$OPENPI_ROOT/.venv/bin/python" - "$DATASET_ROOT" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1]) / "conversion_report.json"
report = json.loads(report_path.read_text())
print(report.get("action_format") or "joint_position_gripper")
PY
)"
fi

"$REPO_ROOT/learning/pi0_ur5e/scripts/install_openpi_config.py" --openpi-root "$OPENPI_ROOT"

LEROBOT_HOME_DIR="$OUTPUT_DIR/lerobot_home"
LINK_PATH="$LEROBOT_HOME_DIR/$LEROBOT_REPO_ID"
mkdir -p "$(dirname "$LINK_PATH")"
rm -f "$LINK_PATH"
ln -s "$DATASET_ROOT" "$LINK_PATH"

export HF_LEROBOT_HOME="$LEROBOT_HOME_DIR"
export PI0_UR5E_LEROBOT_REPO_ID="$LEROBOT_REPO_ID"
export PI0_UR5E_ASSET_ID="$LEROBOT_REPO_ID"
export PI0_UR5E_STATE_DIM="$STATE_DIM"
export PI0_UR5E_TRAIN_STEPS="$STEPS"
export PI0_UR5E_BATCH_SIZE="$BATCH_SIZE"
export PI0_UR5E_ACTION_HORIZON="$ACTION_HORIZON"
if [[ -n "$MAX_TOKEN_LEN" ]]; then
  export PI0_UR5E_MAX_TOKEN_LEN="$MAX_TOKEN_LEN"
fi
export PI0_UR5E_KEEP_PERIOD="$KEEP_PERIOD"
export PI0_UR5E_ACTION_FORMAT="$ACTION_FORMAT"
export PI0_UR5E_LORA="$LORA"
export PI0_UR5E_PI05="$PI05"
if [[ -n "$MODEL_FAMILY" ]]; then
  export PI0_UR5E_MODEL_FAMILY="$MODEL_FAMILY"
  if [[ "$MODEL_FAMILY" == "pi0" ]]; then
    export PI0_UR5E_PI05="false"
  elif [[ "$MODEL_FAMILY" == "pi05" ]]; then
    export PI0_UR5E_PI05="true"
  fi
fi
export PI0_UR5E_USE_DELTA_ACTIONS="$USE_DELTA_ACTIONS"
export PI0_UR5E_FREEZE_MODE="$FREEZE_MODE"
export PI0_UR5E_CAMERA_PADDING="$CAMERA_PADDING"
export PI0_UR5E_DEFAULT_PROMPT="$DEFAULT_PROMPT"
export PI0_UR5E_ASSETS_BASE_DIR="$OUTPUT_DIR/assets"
export PI0_UR5E_CHECKPOINT_BASE_DIR="$OUTPUT_DIR/checkpoints"
case "$XLA_PREALLOCATE" in
  unset|"") unset XLA_PYTHON_CLIENT_PREALLOCATE ;;
  true|false) export XLA_PYTHON_CLIENT_PREALLOCATE="$XLA_PREALLOCATE" ;;
  *) echo "Invalid --xla-preallocate: $XLA_PREALLOCATE (expected true, false, or unset)" >&2; exit 2 ;;
esac
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
if [[ "$WANDB" != "true" ]]; then
  export WANDB_MODE=disabled
fi

COMPUTE_CMD=("$UV_BIN" run scripts/compute_norm_stats.py --config-name "$CONFIG_NAME")
TRAIN_CMD=("$UV_BIN" run scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME")
if [[ "$RESUME" == "true" || "$RESUME" != "false" ]]; then
  TRAIN_CMD+=(--resume)
elif [[ "$OVERWRITE" == "true" ]]; then
  TRAIN_CMD+=(--overwrite)
fi
if [[ "$WANDB" != "true" ]]; then
  TRAIN_CMD+=(--no-wandb-enabled)
fi

echo "OpenPI root: $OPENPI_ROOT"
echo "Dataset root: $DATASET_ROOT"
echo "State dim: $STATE_DIM"
echo "Action horizon: $ACTION_HORIZON"
echo "Max token length: ${MAX_TOKEN_LEN:-default}"
echo "Keep period: $KEEP_PERIOD"
echo "Action format: $ACTION_FORMAT"
echo "Pi0.5 enabled: $PI05"
echo "Model family: ${MODEL_FAMILY:-auto}"
echo "Use delta actions: $USE_DELTA_ACTIONS"
echo "Freeze mode: $FREEZE_MODE"
echo "Compute norm stats command:"
printf 'XLA_PYTHON_CLIENT_PREALLOCATE=%q ' "${XLA_PYTHON_CLIENT_PREALLOCATE:-<unset>}"
printf 'XLA_PYTHON_CLIENT_MEM_FRACTION=%q ' "$XLA_PYTHON_CLIENT_MEM_FRACTION"
printf '%q ' "${COMPUTE_CMD[@]}"
printf '\n'
echo "Train command:"
printf 'XLA_PYTHON_CLIENT_PREALLOCATE=%q ' "${XLA_PYTHON_CLIENT_PREALLOCATE:-<unset>}"
printf 'XLA_PYTHON_CLIENT_MEM_FRACTION=%q ' "$XLA_PYTHON_CLIENT_MEM_FRACTION"
printf '%q ' "${TRAIN_CMD[@]}"
printf '\n'

if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

cd "$OPENPI_ROOT"
"${COMPUTE_CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/compute_norm_stats.log"
"${TRAIN_CMD[@]}" 2>&1 | tee "$OUTPUT_DIR/logs/train.log"
