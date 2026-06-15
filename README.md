# Visuotactile Policy Learning for UR5e

This repository provides an end-to-end system for collecting visuotactile robot
demonstrations, training imitation-learning policies, and deploying the trained
policies on a physical UR5e robot.

The current system supports:

- Meta Quest based single-arm teleoperation and demonstration collection;
- RGB, robot-state, gripper, and optional dual tactile-camera observations;
- Diffusion Policy (DP) training and inference;
- OpenPI `pi0` training and inference;
- real-robot deployment through a shared `run_env.py` interface;
- tactile image rectification, task-specific cropping, marker tracking, and
  headset haptic feedback.

> **Anonymous submission notice**
>
> This repository accompanies a paper that is currently under anonymous
> review. Author names, affiliations, the paper title, and the formal citation
> are intentionally omitted. They will be added after the review process.

## Main Contributions

1. **Visuotactile demonstration collection.** A single-arm UR5e teleoperation
   pipeline combines Meta Quest control, RealSense observations, two tactile
   cameras, robot state, and gripper state. Demonstrations can be stored as H5
   trajectories for downstream policy learning.
2. **DP and pi0 training workflows.** Task-specific scripts under `scripts/`
   cover dataset splitting, DP cache preparation and training, conversion to
   LeRobot format, and `pi0_base` LoRA fine-tuning with or without tactile
   embeddings.
3. **Unified real-robot deployment.** The same environment entry point deploys
   joint-space DP and pi0 policies while retaining camera preprocessing,
   tactile processing, trajectory recording, and configurable safety limits.

## System Overview

```text
Meta Quest controller
        |
        v
run_env.py --agent quest ----------------------+
        |                                      |
        | demonstrations                       | robot commands
        v                                      v
shared/data/bc_data/                     UR5e + gripper
        |
        +-------------------+------------------+
                            |
                 dataset preparation
                            |
              +-------------+-------------+
              |                           |
              v                           v
       Diffusion Policy             LeRobot conversion
       learning/dp/                 learning/pi0_ur5e/
              |                           |
              v                           v
       DP checkpoint                 pi0 LoRA checkpoint
              |                           |
              +-------------+-------------+
                            |
                            v
                    run_env.py --agent
                       dp or pi0
```

## Hardware

The default setup uses:

- one UR5e robot;
- a Robotiq parallel gripper;
- two Intel RealSense cameras;
- two USB tactile cameras;
- a Meta Quest headset and controller for teleoperation;
- an optional separate GPU computer for pi0 policy serving.

The hardware processes communicate with `run_env.py` through ZMQ. Camera and
robot ports, serial numbers, IP addresses, and tactile camera device paths must
be adapted to the local setup.

## Installation

The deployment machine is expected to run Linux. The project has primarily
been used from a Conda environment.

```bash
conda create -n <conda_env_name> python=3.9
conda activate <conda_env_name>
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

Hardware operation additionally requires packages that depend on the local
installation:

```bash
pip install opencv-python pyrealsense2 ur-rtde pyzmq pynput termcolor h5py
```

OpenPI is maintained as a separate checkout and environment. See
[`learning/pi0_ur5e/README.md`](learning/pi0_ur5e/README.md) for its setup.

## 1. Collect Demonstrations

Before operating the robot, verify the robot IP, camera identifiers, workspace,
emergency-stop access, and reset pose.

Start the camera and robot ZMQ nodes:

```bash
python launch_nodes.py
```

In another terminal, start Quest teleoperation and record demonstrations:

```bash
python run_env.py \
  --agent quest \
  --save-data \
  --data-dir ./shared/data/bc_data/<task_name>
```

Tyro command-line options use hyphens. Use `--save-data`, not
`--save_data True`.

Tactile sensing is enabled by default. The initial perspective transforms are
loaded from:

```text
sensor_crop/sensor_config_left.json
sensor_crop/sensor_config_right.json
```

To collect without tactile cameras:

```bash
python run_env.py \
  --agent quest \
  --no-use-tactile \
  --save-data \
  --data-dir ./shared/data/bc_data/<task_name>
```

Detailed instructions:

- [UR5e and Meta Quest teleoperation](Readme/ur5_teleoperation.md)
- [Quick data-collection reference](Readme/QUICK_START.md)
- [Tactile camera setup](Readme/TACTILE_CAMERA_SETUP.md)

## 2. Train Policies

Task-specific SLURM scripts are stored under `scripts/<task_name>/`. Edit the
paths, task name, representation, prompts, and resource requests before
submitting them.

### Diffusion Policy

A typical DP training workflow is:

```bash
sbatch -p gpu scripts/<task_name>/run_split_data.sh
sbatch -p gpu scripts/<task_name>/run_prepare_cache.sh
sbatch -p gpu scripts/<task_name>/run_train_dp.sh
```

For tactile-image training, use the corresponding tactile scripts:

```bash
sbatch -p gpu scripts/<task_name>/run_split_data.sh
sbatch -p gpu scripts/<task_name>/run_prepare_cache_tactile.sh
sbatch -p gpu scripts/<task_name>/run_train_dp_tactile.sh
```

All DP training scripts ultimately invoke `learning/dp/pipeline.py`.

### pi0

The pi0 workflow first converts recorded trajectories to LeRobot format and
then launches OpenPI LoRA training:

```bash
sbatch -p gpu scripts/<task_name>/run_convert_pi0_lerobot.sh
sbatch -p gpu scripts/<task_name>/run_train_pi0_no_tactile.sh
```

For tactile embeddings:

```bash
sbatch -p gpu scripts/<task_name>/run_convert_pi0_lerobot_tactile_emb.sh
sbatch -p gpu scripts/<task_name>/run_train_pi0_tactile_emb.sh
```

The default tactile pi0 representation appends a fixed-size embedding of the
left and right tactile RGB images to `observation.state`.

Detailed training instructions:

- [DP and pi0 experiment workflow](Readme/run_diffusion_policy.md)
- [Complete pi0 conversion, training, and deployment guide](learning/pi0_ur5e/README.md)

## 3. Deploy Trained Policies

Start `launch_nodes.py` before running either policy.

### Diffusion Policy

```bash
python run_env.py \
  --agent dp \
  --dp-ckpt-path <path_to_dp_checkpoint>
```

Add `--save-data --data-dir <output_directory>` to record rollout data.

### pi0

First serve the trained checkpoint on the GPU computer:

```bash
python learning/pi0_ur5e/scripts/serve_policy.py \
  --openpi-root <path_to_openpi> \
  --config-name pi0_ur5e_cup \
  --checkpoint-dir <path_to_pi0_checkpoint> \
  --dataset-root <path_to_converted_lerobot_dataset> \
  --model-family pi0 \
  --port 8000
```

Then connect from the robot computer:

```bash
python run_env.py \
  --agent pi0 \
  --pi0-policy-host <gpu_computer_ip> \
  --pi0-policy-port 8000 \
  --pi0-prompt "<task instruction>" \
  --pi0-state-dim 7
```

For a tactile checkpoint using a 16-dimensional tactile embedding:

```bash
python run_env.py \
  --agent pi0 \
  --pi0-policy-host <gpu_computer_ip> \
  --pi0-policy-port 8000 \
  --pi0-prompt "<task instruction>" \
  --pi0-state-dim 23 \
  --pi0-include-tactile \
  --pi0-tactile-feature-mode image_embedding \
  --pi0-tactile-embedding-dim 16 \
  --use-tactile \
  --safe
```

The action format, state dimension, camera ordering, prompt, tactile mode, and
normalization statistics used for deployment must match training.

## Repository Structure

```text
agents/                 Deployment and teleoperation agents
cameras/                RealSense and OpenCV camera drivers
Data_analysis/          Dataset checking, trimming, export, and visualization
learning/dp/            Diffusion Policy model, dataset, and training pipeline
learning/pi0_ur5e/      LeRobot conversion and OpenPI integration
marker_tracking/        Tactile marker detection and motion visualization
oculus_reader/          Meta Quest controller interface
robots/                 UR5e and gripper interfaces
scripts/                Task-specific data preparation and training jobs
sensor_configs/         Task-specific tactile crop configurations
sensor_crop/            Initial tactile perspective-warp configurations
workflow/               Dataset splitting and workflow utilities
launch_nodes.py         Camera and robot ZMQ server entry point
run_env.py              Collection and real-robot deployment entry point
```

## Data and Checkpoints

Large generated artifacts are intentionally excluded from Git:

```text
shared/
data/
data_split/
outputs/
wandb/
script_results/
```

Do not commit private robot logs, raw participant data, model checkpoints, API
tokens, machine-specific IP addresses, or identifying experiment metadata.

## Safety

This repository controls physical hardware. Before each run:

1. Confirm that the robot is in remote mode and the emergency stop is reachable.
2. Clear the workspace and verify the configured reset pose.
3. Start with conservative control frequency and safety limits.
4. Test perception and policy-server connectivity without enabling motion.
5. Keep an operator ready to stop the robot throughout every rollout.

The `--safe` option limits per-step command changes, but it is not a substitute
for physical safeguards, workspace limits, or human supervision.

## Paper and Citation

This codebase is associated with an anonymous paper submission. During review,
please refer to it as:

```text
Anonymous Authors. "Anonymous Paper Title." Under review.
```

A complete BibTeX entry, paper link, trained models, and release metadata will
be added after the anonymity period.

## Acknowledgements

This project builds on ideas and software from prior robot-learning and
teleoperation systems, including Diffusion Policy, OpenPI, LeRobot, HATO,
MuJoCo, Universal Robots RTDE, and Intel RealSense. Please consult the
corresponding upstream projects and licenses when redistributing derived code
or assets.

## License

Licensing information for the complete repository will be added with the
public release. Third-party components and assets remain subject to their
original licenses.
