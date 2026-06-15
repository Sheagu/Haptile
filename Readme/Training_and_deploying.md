# Training and Deploying Diffusion Policy and pi0

This document describes the workflow used to prepare demonstration data, train
Diffusion Policy (DP) and pi0 models, evaluate checkpoints, and deploy them on
the physical robot.

The examples use generic paths and task names. Replace all values enclosed in
angle brackets before running a command.

## Path Conventions

```bash
export REPO_ROOT=/path/to/project
export OPENPI_ROOT=/path/to/openpi
export TASK_NAME=<task_name>

cd "$REPO_ROOT"
```

The main paths used throughout this guide are:

```text
$REPO_ROOT/shared/data/bc_data/<task_name>   raw demonstration data
$REPO_ROOT/data_split/<task_name>            DP train/test split prefix
$REPO_ROOT/data/<task_name>/ckpts             DP checkpoints
$REPO_ROOT/outputs/                            converted pi0 data and checkpoints
$REPO_ROOT/script_results/                     SLURM output and error logs
```

## End-to-End Workflow

1. Collect demonstrations using `python run_env.py --save-data`.
2. Split the dataset into training and test sets.
3. Train either DP, pi0, or both.
4. Evaluate the trained checkpoints offline.
5. Start the robot hardware nodes and deploy the selected policy.
6. Optionally record policy rollouts for qualitative and quantitative analysis.

## Additional Python Packages

The data processing and experiment logging tools require:

```bash
pip install h5py wandb
```

Install the remaining project dependencies as described in the root
[`README.md`](../README.md).

## Data Validation and Preprocessing

### Check Dataset Integrity

Check timestamp consistency and required fields before training:

```bash
python Data_analysis/check_bc_data_integrity.py \
  shared/data/bc_data/<task_name>
```

Run the same command again after any trimming or conversion step.

### Inspect a sample

Inspect a sample of the processed trajectories:

```bash
python Data_analysis/batch_export_h5_videos.py \
  shared/data/bc_data/<task_name>_trimmed
```

### Crop and Rectify Tactile Images

Create a task-specific crop configuration by selecting four corner points:

```bash
python Data_analysis/crop_tactile_h5_videos.py select-config \
  shared/data/bc_data/<task_name> \
  --config-dir sensor_configs/<task_name> \
  --output-size 320x240
```

Preview the crop:

```bash
python Data_analysis/crop_tactile_h5_videos.py preview \
  shared/data/bc_data/<task_name> \
  --config-dir sensor_configs/<task_name>
```

Convert the complete dataset:

```bash
python Data_analysis/crop_tactile_h5_videos.py convert \
  shared/data/bc_data/<task_name> \
  shared/data/bc_data/<task_name>_tactile_crop \
  --config-dir sensor_configs/<task_name>
```

Inspect an exported H5 trajectory:

```bash
python Data_analysis/test_h5_video_export.py \
  shared/data/bc_data/<task_name>_tactile_crop/<episode>/trajectory.h5
```

### Generate Marker-Tracking Overlays

Export an overlay for one trajectory:

```bash
python Data_analysis/export_marker_tracking_overlay.py \
  shared/data/bc_data/<task_name>_tactile_crop/<episode>/trajectory.h5
```

Replace the tactile videos in an entire dataset with marker-tracking overlays:

```bash
python Data_analysis/batch_replace_tactile_videos_with_marker_overlay.py \
  shared/data/bc_data/<task_name>_tactile_crop
```

If a policy is trained on overlay images, deployment must enable the same
real-time crop and overlay pipeline.

## Running on a SLURM Cluster

Task-specific job scripts are stored under `scripts/<task_name>/`. Update the
project paths, dataset name, resource requests, W&B settings, and task prompt
before submitting a job.

SLURM output and error files are written to a project-local directory:

```text
$REPO_ROOT/script_results/
```

## Diffusion Policy

### Split the Dataset

`workflow/split_data.py` creates train/test directory structures using symbolic
links rather than duplicating all trajectory data.

```bash
python workflow/split_data.py \
  --base_path shared/data/bc_data \
  --output_path data_split \
  --data_name <task_name> \
  --num_trajs 10 25 50
```

Important arguments:

- `--base_path`: parent directory containing the raw task dataset.
- `--output_path`: destination for generated train/test splits.
- `--data_name`: task dataset directory name.
- `--num_trajs`: optional training subset sizes. Multiple values create
  additional directories such as `<task_name>_train_10` and
  `<task_name>_train_25`.

The task-specific equivalent is:

```bash
sbatch -p <partition> scripts/<task_name>/run_split_data.sh
```

### Prepare the Image Cache

DP can store image observations in memory-mapped files to reduce repeated image
decoding during training.

```bash
python learning/dp/pipeline.py \
  --data_path "$REPO_ROOT/data_split/<task_name>" \
  --model_save_path "$REPO_ROOT/data/<task_name>/ckpts/cache_prepare_dummy" \
  --use_train_test_split True \
  --representation_type img-pos \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --batch_size 32 \
  --num_workers 4 \
  --obs_horizon 2 \
  --pred_horizon 16 \
  --action_horizon 8 \
  --num_diffusion_iters 100 \
  --use_memmap_cache True \
  --load_img False \
  --gpu 0 \
  --prepare_cache_only True
```

This command prepares the cache and exits without training a policy. The cache
filename depends on the camera indices, representation, and image-loading
configuration.

Task-specific scripts:

```bash
# RGB/state cache.
sbatch -p <partition> scripts/<task_name>/run_prepare_cache.sh

# RGB/tactile/state cache.
sbatch -p <partition> scripts/<task_name>/run_prepare_cache_tactile.sh
```

### Train a DP Model

Example RGB and joint-state training command:

```bash
python learning/dp/pipeline.py \
  --data_path "$REPO_ROOT/data_split/<task_name>" \
  --model_save_path "$REPO_ROOT/data/<task_name>/ckpts/dp_img_pos_delta" \
  --use_train_test_split True \
  --representation_type img-pos \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --batch_size 32 \
  --num_workers 4 \
  --epochs 300 \
  --eval_freq 10 \
  --save_freq 10 \
  --obs_horizon 2 \
  --pred_horizon 12 \
  --action_horizon 4 \
  --num_diffusion_iters 100 \
  --predict_pos_delta True \
  --image_output_size 64 \
  --color_jitter True \
  --state_noise 0.005 \
  --use_memmap_cache True \
  --load_img False \
  --gpu 0 \
  --use_wandb True \
  --wandb_entity_name <wandb_entity> \
  --wandb_project_name <wandb_project> \
  --wandb_exp_name <experiment_name>
```

Important arguments:

- `--representation_type`: observation modalities separated by `-`, for
  example `img-pos` or `img-tactile_img-pos`.
- `--camera_indices`: camera indices included in the visual observation.
- `--obs_horizon`: number of observation frames provided to the policy.
- `--pred_horizon`: number of future action steps predicted per diffusion
  sample.
- `--action_horizon`: number of predicted steps used before replanning.
- `--predict_pos_delta`: predict joint-position changes instead of absolute
  joint targets.
- `--use_memmap_cache`: use the prepared image cache.
- `--load_img False`: avoid loading the entire image dataset into RAM.

Task-specific scripts:

```bash
sbatch -p <partition> scripts/<task_name>/run_train_dp.sh
sbatch -p <partition> scripts/<task_name>/run_train_dp_tactile.sh
```

### DP Training Outputs

A DP experiment directory normally contains:

- `last.ckpt`: final non-EMA model parameters.
- `ema_last.ckpt`: exponential-moving-average parameters corresponding to
  `last.ckpt`.
- `model_epoch_<epoch>.ckpt`: periodic non-EMA checkpoints.
- `ema_model_epoch_<epoch>.ckpt`: periodic EMA checkpoints.
- `stats.pkl`: normalization statistics required during deployment.
- `args_log.txt` or `args_log.json`: training configuration.
- `full_eval_summary.pkl`: aggregate offline evaluation results, when enabled.

Keep checkpoints, normalization statistics, and argument logs together.
Deployment reconstructs the model configuration from files in the checkpoint
directory.

## Offline DP Evaluation

### Evaluate One Trajectory

```bash
python learning/dp/pipeline.py \
  --eval True \
  --load_path <checkpoint_path> \
  --eval_path "$REPO_ROOT/data_split/<task_name>_test/<episode>" \
  --save_path "$REPO_ROOT/data/<task_name>/eval_results/<result_name>" \
  --representation_type img-pos \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --obs_horizon 2 \
  --pred_horizon 12 \
  --action_horizon 4 \
  --num_diffusion_iters 100 \
  --predict_pos_delta True \
  --image_output_size 64 \
  --gpu 0
```

The architecture and preprocessing arguments must match the training run.

### Evaluate a Complete Test Set

`eval_dir.py` reads the saved training arguments next to the checkpoint, which
reduces configuration mismatches:

```bash
python eval_dir.py \
  --ckpt_path <checkpoint_path> \
  --eval_dir "$REPO_ROOT/data_split/<task_name>_test" \
  --save_path "$REPO_ROOT/data/<task_name>/eval_results/<result_file>.pkl"
```

The output contains per-trajectory and aggregate errors such as action MSE.

## Deploy DP on the Robot

Start the hardware nodes:

```bash
python launch_nodes.py
```

In another terminal:

```bash
python run_env.py \
  --agent dp \
  --dp-ckpt-path <checkpoint_path> \
  --hz 15 \
  --safe
```

To record policy rollouts:

```bash
python run_env.py \
  --agent dp \
  --dp-ckpt-path <checkpoint_path> \
  --hz 15 \
  --safe \
  --save-data \
  --data-dir ./shared/data/bc_data/<task_name>_dp_rollouts
```

`--safe` enables the per-step action limits configured by
`--safe-max-joint-delta` and `--safe-max-hand-delta`. It does not replace
physical safety measures or operator supervision.

### Deploy a DP Model Trained on Marker Overlays

When tactile training videos were replaced with marker-tracking overlays,
generate the same representation during deployment:

```bash
python run_env.py \
  --agent dp \
  --dp-ckpt-path <checkpoint_path> \
  --hz 15 \
  --safe \
  --use-tactile \
  --enable-marker-tracking \
  --tactile-crop-config-dir sensor_configs/<task_name> \
  --use-marker-tracking-overlay-for-policy
```

`--tactile-crop-config-dir` applies the task-specific tactile crop before the
marker overlay is generated. Use `--tactile-crop-input-width` and
`--tactile-crop-input-height` if the live intermediate image resolution differs
from the resolution assumed by the crop configuration.

## pi0

The pi0 workflow uses a separate OpenPI checkout and environment. For a more
detailed guide, see [`learning/pi0_ur5e/README.md`](../learning/pi0_ur5e/README.md).

### Set Up OpenPI

Example setup inside the OpenPI checkout:

```bash
cd "$OPENPI_ROOT"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install --python .venv/bin/python -e .
```

If the raw dataset uses different field names, update:

```text
$REPO_ROOT/learning/pi0_ur5e/configs/dataset_schema.yaml
```

### Convert Data Without Tactile Embeddings

Run conversion from an environment that provides LeRobot:

```bash
cd "$OPENPI_ROOT"

uv run python "$REPO_ROOT/learning/pi0_ur5e/scripts/convert_to_lerobot.py" \
  --input-root "$REPO_ROOT/shared/data/bc_data/<task_name>" \
  --output-root "$REPO_ROOT/outputs/<task_name>_lerobot_no_tactile" \
  --config "$REPO_ROOT/learning/pi0_ur5e/configs/dataset_schema.yaml" \
  --task-name <task_name> \
  --repo-id local/pi0_ur5e_<task_name>_no_tactile \
  --default-prompt "<task instruction>" \
  --action-mode joint_position_gripper \
  --include-tactile false \
  --overwrite true
```

The resulting `observation.state` is normally 7D:

```text
[joint_0, ..., joint_5, gripper]
```

### Convert Data With Tactile Embeddings

```bash
cd "$OPENPI_ROOT"

uv run python "$REPO_ROOT/learning/pi0_ur5e/scripts/convert_to_lerobot.py" \
  --input-root "$REPO_ROOT/shared/data/bc_data/<task_name>" \
  --output-root "$REPO_ROOT/outputs/<task_name>_lerobot_tactile_emb" \
  --config "$REPO_ROOT/learning/pi0_ur5e/configs/dataset_schema.yaml" \
  --task-name <task_name> \
  --repo-id local/pi0_ur5e_<task_name>_tactile_emb \
  --default-prompt "<task instruction>" \
  --action-mode joint_position_gripper \
  --include-tactile true \
  --tactile-feature-mode image_embedding \
  --tactile-embedding-dim 16 \
  --overwrite true
```

With a 16D tactile embedding, the converted state is normally:

```text
robot state (7D) + tactile embedding (16D) = observation.state (23D)
```

Task-specific conversion scripts are also available:

```bash
sbatch -p <partition> scripts/<task_name>/run_convert_pi0_lerobot.sh
sbatch -p <partition> scripts/<task_name>/run_convert_pi0_lerobot_tactile_emb.sh
```

### Train pi0 With LoRA

The helper script `learning/pi0_ur5e/scripts/train_pi0_base.sh`:

1. installs the project-specific UR5e configuration into OpenPI;
2. links the converted dataset into a local `HF_LEROBOT_HOME`;
3. reads `observation.state.shape` from `meta/info.json`;
4. computes normalization statistics;
5. starts OpenPI training.

For `pi0_base` LoRA fine-tuning, use:

```text
--model-family pi0 --pi05 false --lora true
```

The following example trains a no-tactile model:

```bash
cd "$REPO_ROOT"

bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root "$REPO_ROOT/outputs/<task_name>_lerobot_no_tactile" \
  --output-dir "$REPO_ROOT/outputs/pi0_<task_name>_no_tactile_lora" \
  --openpi-root "$OPENPI_ROOT" \
  --repo-id local/pi0_ur5e_<task_name>_no_tactile \
  --exp-name <task_name>_pi0_base_no_tactile_lora \
  --steps 30000 \
  --batch-size 16 \
  --model-family pi0 \
  --pi05 false \
  --lora true \
  --camera-padding-strategy zeros \
  --use-delta-actions true \
  --include-tactile false \
  --default-prompt "<task instruction>" \
  --dry-run false
```

For tactile training, use the tactile-converted dataset and set
`--include-tactile true`:

```bash
cd "$REPO_ROOT"

bash learning/pi0_ur5e/scripts/train_pi0_base.sh \
  --dataset-root "$REPO_ROOT/outputs/<task_name>_lerobot_tactile_emb" \
  --output-dir "$REPO_ROOT/outputs/pi0_<task_name>_tactile_emb_lora" \
  --openpi-root "$OPENPI_ROOT" \
  --repo-id local/pi0_ur5e_<task_name>_tactile_emb \
  --exp-name <task_name>_pi0_base_tactile_emb_lora \
  --steps 30000 \
  --batch-size 16 \
  --model-family pi0 \
  --pi05 false \
  --lora true \
  --camera-padding-strategy zeros \
  --use-delta-actions true \
  --include-tactile true \
  --default-prompt "<task instruction>" \
  --dry-run false
```

Task-specific training scripts:

```bash
sbatch -p <partition> scripts/<task_name>/run_train_pi0_no_tactile.sh
sbatch -p <partition> scripts/<task_name>/run_train_pi0_tactile_emb.sh
```

`--use-delta-actions true` applies the OpenPI action transform during training.
The converted dataset can still store actions in
`joint_position_gripper` format.

## Serve a pi0 Checkpoint

Use the same converted dataset variant used during training. The serving helper
reads the dataset state dimension and configures the policy accordingly.

### No-Tactile Checkpoint

```bash
cd "$REPO_ROOT"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root "$OPENPI_ROOT" \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir <pi0_checkpoint_directory> \
  --dataset-root "$REPO_ROOT/outputs/<task_name>_lerobot_no_tactile" \
  --model-family pi0 \
  --use-delta-actions true \
  --camera-padding-strategy zeros \
  --default-prompt "<task instruction>" \
  --port 8000
```

### Tactile-Embedding Checkpoint

```bash
cd "$REPO_ROOT"

XLA_PYTHON_CLIENT_PREALLOCATE=false \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root "$OPENPI_ROOT" \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir <pi0_checkpoint_directory> \
  --dataset-root "$REPO_ROOT/outputs/<task_name>_lerobot_tactile_emb" \
  --model-family pi0 \
  --use-delta-actions true \
  --camera-padding-strategy zeros \
  --default-prompt "<task instruction>" \
  --port 8000
```

## Deploy pi0 on the Robot

Start `launch_nodes.py` on the robot computer before deploying the policy.

### No-Tactile Policy

```bash
python run_env.py \
  --agent pi0 \
  --pi0-policy-host <policy_server_ip> \
  --pi0-policy-port 8000 \
  --pi0-prompt "<task instruction>" \
  --pi0-state-dim 7
```

### Tactile-Embedding Policy

For a 16D tactile embedding:

```bash
python run_env.py \
  --agent pi0 \
  --pi0-policy-host <policy_server_ip> \
  --pi0-policy-port 8000 \
  --pi0-prompt "<task instruction>" \
  --pi0-state-dim 23 \
  --pi0-include-tactile \
  --pi0-tactile-feature-mode image_embedding \
  --pi0-tactile-embedding-dim 16 \
  --use-tactile
```

The deployment state dimension must match the converted training dataset.

### pi0 With Real-Time Marker Overlays

If the tactile embeddings were generated from cropped marker-overlay images,
enable the same preprocessing during deployment:

```bash
python run_env.py \
  --agent pi0 \
  --pi0-policy-host <policy_server_ip> \
  --pi0-policy-port 8000 \
  --pi0-prompt "<task instruction>" \
  --pi0-state-dim 23 \
  --pi0-include-tactile \
  --pi0-tactile-feature-mode image_embedding \
  --pi0-tactile-embedding-dim 16 \
  --use-tactile \
  --enable-marker-tracking \
  --tactile-crop-config-dir sensor_configs/<task_name> \
  --use-marker-tracking-overlay-for-policy
```

In this mode, both the pi0 tactile inputs and the marker-tracking display
windows use the task-cropped overlay images.

## Deployment Consistency Checklist

Before operating the robot, verify that:

- camera ordering matches the training dataset;
- DP representation type and horizons match the checkpoint configuration;
- the DP checkpoint directory contains its normalization statistics;
- the pi0 action format matches the converted dataset;
- `pi0_state_dim` matches `observation.state.shape`;
- tactile embedding mode and dimension match conversion and training;
- the prompt is consistent with the training data;
- crop and marker-overlay preprocessing match the training representation;
- the policy server is reachable before robot motion is enabled.