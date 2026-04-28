# UR5e pi0_base Fine-Tuning

This folder adds an isolated adapter for fine-tuning OpenPI `pi_0_base` / `pi0_base` on UR5e data with one wrist camera, one third-person/base camera, and optional tactile low-dimensional features.

The code is intentionally kept under `learning/pi0_ur5e` and does not modify robot control, camera, or data collection modules. Existing repository data is detected from per-frame `.pkl` folders or `trajectory.h5` files written by `writers/h5_trajectory_writer.py`.

## Environment

Install the current repository dependencies first, then install optional conversion/checking tools:

```bash
pip install -r requirements.txt
pip install h5py pyyaml opencv-python matplotlib pytest pyzmq msgpack websockets
```

Install OpenPI in a separate checkout following its official instructions. Keep Python, CUDA, and JAX versions consistent with that checkout. This repository injects a small `pi0_ur5e_cup` config patch into `openpi/src/openpi/training/config.py` when running validation/training; it does not vendor OpenPI source.

Recommended local layout:

```bash
/home/rpl/yongqiang/tele-gsy   # this repository
/home/rpl/yongqiang/openpi     # OpenPI checkout with .venv managed by uv
```

## Data Preparation

Raw input can be one of:

- a directory containing episode subfolders with per-frame `.pkl` files,
- a directory containing `trajectory.h5` files,
- a single `.npz` episode,
- a real LeRobot v2 dataset created by this adapter.

Edit `learning/pi0_ur5e/configs/dataset_schema.yaml` if your field names differ. The default detector looks for current repo keys such as `base_rgb`, `wrist_rgb`, `joint_positions`, `ee_pos_quat`, `gripper_position`, `touch`, and `control`.

Convert to the LeRobot/OpenPI-style layout:

```bash
cd /home/rpl/yongqiang/openpi
uv run python /home/rpl/yongqiang/tele-gsy/learning/pi0_ur5e/scripts/convert_to_lerobot.py \
  --input-root /path/to/raw_dataset \
  --output-root /path/to/lerobot_dataset \
  --config /home/rpl/yongqiang/tele-gsy/learning/pi0_ur5e/configs/dataset_schema.yaml \
  --task-name cup_pick_place \
  --repo-id local/pi0_ur5e_cup \
  --default-prompt "pick up the paper cup and place it on the target" \
  --include-tactile false \
  --overwrite true
```

Output is a real LeRobot v2 dataset with `meta/`, `data/*.parquet`, embedded images, and `conversion_report.json`. Images are resized to `224x224`. State and action are saved as `float32`. The default action mode is `ee_delta_6d_gripper` with `[dx, dy, dz, droll, dpitch, dyaw, gripper]`.

If OpenPI expects a third camera slot, `camera_padding_strategy` defaults to `duplicate_base`. Other options are `none`, `duplicate_wrist`, and `zeros`.

## Sanity Check

```bash
python learning/pi0_ur5e/scripts/inspect_dataset.py \
  --dataset-root /path/to/raw_or_lerobot_dataset \
  --config learning/pi0_ur5e/configs/dataset_schema.yaml \
  --output-dir outputs/pi0_dataset_inspection
```

Review `inspection_report.json`, `conversion_report.json`, and images in `debug_vis/`. Common issues:

- large timestamp gaps mean dropped frames or mixed episode files,
- low action/state delta correlation can indicate the wrong action frame or absolute-vs-delta mismatch,
- gripper close events should line up with frames where the gripper is near the cup,
- NaN/Inf or unusually large tactile values should be fixed before training.

Validate the converted dataset with OpenPI's actual LeRobot loader:

```bash
python learning/pi0_ur5e/scripts/validate_lerobot_with_openpi.py \
  --dataset-root /path/to/lerobot_dataset \
  --openpi-root /home/rpl/yongqiang/openpi \
  --config-name pi0_ur5e_cup \
  --expected-action-dim 7
```

## Training

The training script injects `learning/pi0_ur5e/openpi_patches/pi0_ur5e_cup_config.py` into OpenPI and runs the official OpenPI commands:

```bash
cd /home/rpl/yongqiang/openpi
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/compute_norm_stats.py --config-name pi0_ur5e_cup
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_ur5e_cup --exp-name ur5e_cup_pi0_base --overwrite
```

3000-step sanity check from this repository:

```bash
bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root /path/to/lerobot_dataset \
  --output-dir outputs/pi0_ur5e_cup \
  --openpi-root /home/rpl/yongqiang/openpi \
  --steps 3000 \
  --batch-size 16 \
  --lora true
```

Continue to 10000 steps:

```bash
bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root /path/to/lerobot_dataset \
  --output-dir outputs/pi0_ur5e_cup_10k \
  --openpi-root /home/rpl/yongqiang/openpi \
  --steps 10000 \
  --batch-size 16 \
  --lora true \
  --resume true
```

LoRA is recommended for the first pass because it fits a 24GB RTX 3090 class GPU. Full fine-tuning may improve final performance but needs much more memory. The config keeps the pi0_base model action dimension for checkpoint compatibility and pads the 7D UR5e actions internally; policy outputs are sliced back to 7D.

## Evaluation And Inference

Offline replay plots:

```bash
python learning/pi0_ur5e/scripts/eval_replay_dataset.py \
  --dataset-root /path/to/lerobot_dataset \
  --output-dir outputs/pi0_replay_eval
```

Policy server:

```bash
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root /home/rpl/yongqiang/openpi \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir outputs/pi0_ur5e_cup/checkpoints/pi0_ur5e_cup/ur5e_cup_pi0_base/2999 \
  --dataset-root /path/to/lerobot_dataset \
  --port 8000
```

Two-computer deployment:

Run this on the GPU computer:

```bash
cd /home/rpl/yongqiang/tele-gsy
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root /home/rpl/yongqiang/openpi \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir /path/to/checkpoint_step_dir \
  --dataset-root /path/to/lerobot_dataset \
  --port 8000
```

Run this on the robot/control computer to verify the network path before connecting the real robot:

```bash
cd /home/rpl/yongqiang/tele-gsy
python learning/pi0_ur5e/scripts/query_policy_server.py \
  --host <gpu_computer_ip> \
  --port 8000
```

The robot-side adapter should send observations with these raw keys:

```python
from pi0_ur5e.policy_client import WebsocketPolicyClient, build_policy_observation, safe_action
from pi0_ur5e.schema import Pi0Ur5eConfig

cfg = Pi0Ur5eConfig()
policy = WebsocketPolicyClient("<gpu_computer_ip>", 8000)

obs = build_policy_observation(
    base_rgb=base_rgb_uint8_224x224x3,
    wrist_rgb=wrist_rgb_uint8_224x224x3,
    state=robot_state_float32_7d,
    prompt=cfg.default_prompt,
)
action = safe_action(policy, obs, cfg.action_limits)
```

`policy.action_chunk(obs)` returns the full OpenPI action horizon. `policy.act(obs)` returns the first action in that chunk for simple step-by-step control. Keep the existing action clipping and workspace checks on the robot/control computer, even though inference runs remotely.

Dry-run rollout:

```bash
python learning/pi0_ur5e/scripts/rollout_pi0_policy.py \
  --policy /path/to/exported_policy.pkl \
  --output-dir outputs/pi0_dry_run \
  --steps 10
```

Network dry-run rollout against the GPU server:

```bash
python learning/pi0_ur5e/scripts/rollout_pi0_policy.py \
  --server-host <gpu_computer_ip> \
  --server-port 8000 \
  --output-dir outputs/pi0_network_dry_run \
  --steps 10
```

Real robot loop with the existing deployment entrypoints:

```bash
# Robot/control computer, terminal 1: robot and camera ZMQ nodes
python launch_nodes.py

# Robot/control computer, terminal 2: existing environment loop using remote pi0
python run_env.py \
  --agent pi0_eef \
  --pi0-policy-host <gpu_computer_ip> \
  --pi0-policy-port 8000
```

Use `pi0_eef` for the current `pi0_ur5e_cup` policy because it was trained with `ee_delta_6d_gripper` actions. The plain `pi0` branch is present for future joint-space pi0 checkpoints, but it will reject the current EEF-delta action shape.

Real robot execution is intentionally disabled in the generic adapter. A project-specific wrapper must verify action clipping, workspace bounds, emergency stop, max step delta, and gripper range before enabling `--allow-real-robot true`.

## Real Training Order

```bash
# 1. convert
cd /home/rpl/yongqiang/openpi
uv run python /home/rpl/yongqiang/tele-gsy/learning/pi0_ur5e/scripts/convert_to_lerobot.py \
  --input-root /path/to/raw_dataset \
  --output-root /path/to/lerobot_dataset \
  --config /home/rpl/yongqiang/tele-gsy/learning/pi0_ur5e/configs/dataset_schema.yaml \
  --overwrite true

# 2. inspect
cd /home/rpl/yongqiang/tele-gsy
python learning/pi0_ur5e/scripts/inspect_dataset.py \
  --dataset-root /path/to/lerobot_dataset \
  --output-dir outputs/pi0_dataset_inspection

# 3. validate with OpenPI loader
python learning/pi0_ur5e/scripts/validate_lerobot_with_openpi.py \
  --dataset-root /path/to/lerobot_dataset \
  --openpi-root /home/rpl/yongqiang/openpi

# 4-5. compute_norm_stats and train
bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root /path/to/lerobot_dataset \
  --output-dir outputs/pi0_ur5e_cup \
  --openpi-root /home/rpl/yongqiang/openpi \
  --steps 3000 \
  --batch-size 16 \
  --lora true

# 6. serve trained policy
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root /home/rpl/yongqiang/openpi \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir outputs/pi0_ur5e_cup/checkpoints/pi0_ur5e_cup/ur5e_cup_pi0_base/2999 \
  --dataset-root /path/to/lerobot_dataset
```

## Safety

Default scripts do not command the UR5e. Validate in this order: dataset inspection, dry-run rollout, low-speed action logging, then real task execution with an operator at the emergency stop. Keep translation/rotation deltas clipped and enforce workspace bounds.

## Tactile

First version does not feed raw tactile images to pi0. `tactile_features.py` converts tactile arrays to low-dimensional features such as contact flag, normal/tangential force proxies, slip score proxy, pressure means, pressure sum, and pressure center. These features are appended to `observation.state` only when `include_tactile=true` and `tactile_feature_mode=low_dim`.

Base pi0 has no tactile pretraining, so compare no-tactile and tactile runs before assuming tactile helps.

## FAQ

- Fails to approach cup: check action frame, delta-vs-absolute mode, normalization stats, camera mapping, and timestamp sync.
- Gripper closes early/late: inspect timestamp order and `gripper_position`/action alignment.
- Loss looks normal but rollout is poor: check normalization and action denormalization in OpenPI.
- Tactile makes performance worse: train a no-tactile baseline first, then run tactile ablations.
