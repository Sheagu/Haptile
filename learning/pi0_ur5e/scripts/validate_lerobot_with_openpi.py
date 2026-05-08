#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import textwrap


VALIDATOR = r"""
import os
import sys
import numpy as np

dataset_root = sys.argv[1]
config_name = sys.argv[2]
expected_action_dim = int(sys.argv[3])
expected_state_dim = None if sys.argv[4] == "None" else int(sys.argv[4])

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from openpi.training import config as openpi_config
from openpi.training import data_loader as openpi_data_loader

metadata = LeRobotDatasetMetadata(repo_id=dataset_root, root=dataset_root)
dataset = LeRobotDataset(repo_id=dataset_root, root=dataset_root)
sample = dataset[0]
print("LeRobot sample keys:", sorted(sample.keys()))

def shape_of(key):
    value = sample[key]
    shape = tuple(value.shape) if hasattr(value, "shape") else np.asarray(value).shape
    print(f"{key}: shape={shape}, dtype={getattr(value, 'dtype', type(value))}")
    return shape

base_shape = shape_of("observation.images.base_rgb")
wrist_shape = shape_of("observation.images.wrist_rgb")
state_shape = shape_of("observation.state")
action_shape = shape_of("action")
if action_shape[-1] != expected_action_dim:
    raise SystemExit(f"Expected action_dim={expected_action_dim}, got {action_shape[-1]}")
if expected_state_dim is not None and state_shape[-1] != expected_state_dim:
    raise SystemExit(f"Expected state_dim={expected_state_dim}, got {state_shape[-1]}")

cfg = openpi_config.get_config(config_name)
cfg_data = cfg.data
if getattr(cfg_data, "expected_state_dim", None) not in {None, state_shape[-1]}:
    raise SystemExit(
        f"OpenPI config expected_state_dim={cfg_data.expected_state_dim}, dataset state_dim={state_shape[-1]}"
    )
data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
raw_dataset = openpi_data_loader.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
transformed = openpi_data_loader.transform_dataset(raw_dataset, data_config, skip_norm_stats=True)
openpi_sample = transformed[0]
print("OpenPI transformed sample keys:", sorted(openpi_sample.keys()))
print("OpenPI image keys:", sorted(openpi_sample["image"].keys()))
print("OpenPI state shape:", np.asarray(openpi_sample["state"]).shape)
print("OpenPI actions shape:", np.asarray(openpi_sample["actions"]).shape)
if np.asarray(openpi_sample["actions"]).shape[-1] != cfg.model.action_dim:
    raise SystemExit("OpenPI model transform did not pad actions to model action_dim")
print("Validation OK")
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a converted LeRobot dataset with OpenPI's real loader.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--openpi-root", required=True, type=Path)
    parser.add_argument("--config-name", default="pi0_ur5e_cup")
    parser.add_argument("--expected-action-dim", type=int, default=7)
    parser.add_argument("--expected-state-dim", type=int, default=None)
    parser.add_argument("--camera-padding-strategy", default="zeros")
    parser.add_argument("--model-family", default=None, choices=["pi0", "pi05", "pi0_fast"])
    parser.add_argument("--use-delta-actions", default="true")
    parser.add_argument("--freeze-mode", default="default", choices=["default", "vision_action_head", "action_head"])
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("install_openpi_config.py")),
            "--openpi-root",
            str(args.openpi_root),
        ],
        check=True,
    )
    env = os.environ.copy()
    env.setdefault("HF_DATASETS_CACHE", "/tmp/hf_datasets_cache")
    env.update(
        {
            "PI0_UR5E_LEROBOT_REPO_ID": str(args.dataset_root.resolve()),
            "PI0_UR5E_ASSET_ID": "pi0_ur5e_cup",
            "PI0_UR5E_CAMERA_PADDING": args.camera_padding_strategy,
            "PI0_UR5E_USE_DELTA_ACTIONS": str(args.use_delta_actions).lower(),
            "PI0_UR5E_FREEZE_MODE": args.freeze_mode,
        }
    )
    if args.model_family is not None:
        env["PI0_UR5E_MODEL_FAMILY"] = args.model_family
    if args.expected_state_dim is not None:
        env["PI0_UR5E_STATE_DIM"] = str(args.expected_state_dim)
    code = textwrap.dedent(VALIDATOR)
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "-c",
            code,
            str(args.dataset_root.resolve()),
            args.config_name,
            str(args.expected_action_dim),
            str(args.expected_state_dim),
        ],
        cwd=args.openpi_root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
