# Diffusion Policy Training Workflow

`learning/dp` implements the Diffusion Policy workflow for data loading, image preprocessing, normalization statistics, training, checkpoint saving, and offline evaluation. This document describes the full training sequence: data splitting, data cache generation, and training.

## 1. Data Directory Convention

It is recommended to place the raw data under one task directory, for example:

```text
/path/to/cup_demo/
  20260501_120000/
    trajectory.h5
  20260501_120530/
    trajectory.h5
  ...
```

Each episode is a subdirectory. The DP loader supports two formats:

- `trajectory.h5`: the recommended format. It contains numeric data under `frames` and optional image streams under `videos`.
- Per-frame `.pkl`: the legacy format, with one pickle file per frame. If a pickle does not contain images, the loader will try to read matching camera images such as `-0.png`, `-1.png`, and so on.

The training script expects split directories by default:

```text
/path/to/cup_demo_train/
/path/to/cup_demo_test/
```

If `--data_path /path/to/cup_demo` and `--use_train_test_split True` are provided, the script automatically reads `/path/to/cup_demo_train` as the training set and `/path/to/cup_demo_test` as the test set.

## 2. Data Split

If the train/test directories do not exist yet, use `workflow/split_data.py` to generate them in one step. This script does not copy or move the original episodes. It creates symbolic links in the output directory:

```bash
python workflow/split_data.py \
  --base_path /path/to/raw_data \
  --output_path /path/to/split_data \
  --data_name cup_demo \
  --num_trajs 10 25 50
```

Assuming the raw data is stored in `/path/to/raw_data/cup_demo`, the command above creates:

```text
/path/to/split_data/cup_demo_train/
/path/to/split_data/cup_demo_test/
/path/to/split_data/cup_demo_train_10/
/path/to/split_data/cup_demo_train_25/
/path/to/split_data/cup_demo_train_50/
```

Default split behavior:

- Episodes whose directory names end with `failed`, `ood`, `ikbad`, `heated`, `stop`, or `hard` are filtered out.
- The remaining episodes are sorted by directory name.
- The last episode is linked into `<data_name>_test`.
- All other episodes are linked into `<data_name>_train`.
- `--num_trajs 10 25 50` additionally samples subsets of the corresponding sizes from `<data_name>_train` and creates `<data_name>_train_10`, `<data_name>_train_25`, and `<data_name>_train_50`.

During training, pass the split path without the `_train` or `_test` suffix to `--data_path`:

```bash
--data_path /path/to/split_data/cup_demo
```

To train on a subset such as `cup_demo_train_10`, add:

```bash
--train_suffix _10
```

The training script will then read `/path/to/split_data/cup_demo_train_10` as the training set, while still reading `/path/to/split_data/cup_demo_test` as the test set.

The legacy `workflow/create_eval.py` can also manually move episodes from one directory into a test directory, but it modifies the input directory and is usually not the preferred workflow.

Filtering options during training:

- `--traj_type plain`: the default. Excludes episodes whose directory names end with `failed`, `ood`, `ikbad`, `heated`, `stop`, or `hard`.
- `--traj_type all`: disables filtering and uses all episodes for training/testing.
- `--prefix 0-1`: uses only episodes whose directory names start with `0` or `1`. Omit this option if prefix filtering is not needed.

## 3. Generate the Data Cache

When image data is large, it is recommended to generate a memmap cache first. Training can then read preprocessed images directly from the cache instead of repeatedly decoding H5 videos or loading image files.

```bash
python learning/dp/pipeline.py \
  --data_path /path/to/split_data/cup_demo \
  --representation_type img-depth-eef-pos-touch \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --touch_dim 60 \
  --obs_horizon 1 \
  --pred_horizon 16 \
  --action_horizon 8 \
  --batch_size 32 \
  --num_workers 16 \
  --use_train_test_split True \
  --use_memmap_cache True \
  --prepare_cache_only True \
  --model_save_path /tmp/dp_cache_prepare
```

The default cache paths are generated automatically:

```text
/path/to/split_data/cup_demo_train/01-True-mem.dat
/path/to/split_data/cup_demo_test/01-True-mem.dat
```

Here, `01` comes from `--camera_indices 01`, and `True` means `representation_type` includes `depth`. If RGB only is used and `depth` is not included, the file name will be `01-False-mem.dat`.

You can also explicitly specify the training cache file:

```bash
python learning/dp/pipeline.py \
  --data_path /path/to/split_data/cup_demo \
  --representation_type img-depth-eef-pos-touch \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --touch_dim 60 \
  --use_memmap_cache True \
  --memmap_loader_path /path/to/split_data/cup_demo_train/camera01_rgbd_mem.dat \
  --prepare_cache_only True \
  --model_save_path /tmp/dp_cache_prepare
```

When a cache path is specified explicitly, the test cache uses the same file name under the `_test` directory, for example `/path/to/split_data/cup_demo_test/camera01_rgbd_mem.dat`.

## 4. Start Training

After the cache is ready, train with the same data and model parameters:

```bash
python learning/dp/pipeline.py \
  --data_path /path/to/split_data/cup_demo \
  --representation_type img-depth-eef-pos-touch \
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
  --model_save_path /path/to/dp_ckpts
```

By default, the training output directory appends a timestamp and key parameter suffixes under `--model_save_path`. Main files in the run directory:

- `args.txt`: command-line arguments for this training run.
- `stats.pkl`: normalization statistics computed from the training set. This file must be in the same directory as the model when loading a checkpoint.
- `last.ckpt`: the final checkpoint.
- `ema_last.ckpt`: EMA weights, usually used for inference/evaluation.
- `model_epoch_<N>.ckpt` and `ema_model_epoch_<N>.ckpt`: intermediate checkpoints saved according to `--save_freq`.
- `full_eval_summary.pkl`: offline evaluation summary over all test episodes in the `_test` directory after training finishes.

## 5. Common Configurations

Single-arm UR5e + Robotiq, two RGBD cameras, using EEF, joint position, and touch:

```bash
--representation_type img-depth-eef-pos-touch \
--camera_indices 01 \
--joint_state_dim 7 \
--action_dim 7 \
--eef_dim 6
```

State only, without images:

```bash
--representation_type eef-pos-touch \
--use_memmap_cache False
```

RGB only, without depth:

```bash
--representation_type img-eef-pos-touch \
--camera_indices 01
```

Legacy dual-arm data usually uses 24D action/state and 12D EEF:

```bash
--representation_type img-depth-eef-hand_pos-touch \
--camera_indices 012 \
--joint_state_dim 24 \
--action_dim 24 \
--eef_dim 12 \
--hand_pos_dim 12
```

## 6. Evaluate a Checkpoint Separately

Offline evaluation on a single episode:

```bash
python learning/dp/pipeline.py \
  --eval True \
  --load_path /path/to/dp_ckpts/<run>/ema_last.ckpt \
  --eval_path /path/to/cup_demo_test/20260501_120530 \
  --save_path /path/to/eval_outputs \
  --representation_type img-depth-eef-pos-touch \
  --camera_indices 01 \
  --joint_state_dim 7 \
  --action_dim 7 \
  --eef_dim 6 \
  --touch_dim 60
```

The directory containing `--load_path` must also contain the corresponding `stats.pkl`.

## 7. Troubleshooting

- Data not found: check whether `--use_train_test_split True` is enabled. When enabled, the script reads `<data_path>_train` and `<data_path>_test`, not `<data_path>` itself.
- Cache not used: make sure the training command and cache-generation command use the same `--representation_type`, `--camera_indices`, and depth setting.
- H5 images are decoded every time: enable `--use_memmap_cache True` and run `--prepare_cache_only True` once before training.
- Dimension mismatch: check that `--action_dim`, `--joint_state_dim`, `--eef_dim`, and `--touch_dim` match `control`, `joint_positions`, `ee_pos_quat`, and `touch` in the data.
- When the test set contains multiple episodes, the script rotates through eval trajectories. Multiple test episodes with different lengths do not share one eval memmap cache; this is the expected behavior of the current implementation.
