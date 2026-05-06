from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import write_json


PI0_IMAGE_KEY_MAP = {
    "base_rgb": "observation.images.base_rgb",
    "wrist_rgb": "observation.images.wrist_rgb",
    "third_rgb": "observation.images.third_rgb",
}


def build_openpi_config_patch(
    *,
    dataset_root: str,
    checkpoint: str = "pi0_base",
    include_tactile: bool = False,
    action_format: str = "joint_position_gripper",
    action_dim: int = 7,
    state_dim: int | None = None,
    camera_padding_strategy: str = "duplicate_base",
) -> dict[str, Any]:
    return {
        "model": {"checkpoint": checkpoint, "action_dim": action_dim},
        "dataset": {
            "type": "lerobot",
            "root": dataset_root,
            "image_keys": ["observation.images.base_rgb", "observation.images.wrist_rgb"],
            "state_key": "observation.state",
            "action_key": "action",
            "prompt_key": "language_instruction",
            "camera_padding_strategy": camera_padding_strategy,
        },
        "robot": {
            "type": "ur5e",
            "action_format": action_format,
            "state_contains_tactile": include_tactile,
            "state_dim": state_dim,
        },
        "notes": [
            "This is an adapter patch. Do not copy OpenPI source into this repo.",
            "If your OpenPI checkout requires a Python config class, import these values or symlink a small wrapper into openpi/src/openpi/training/config.py.",
        ],
    }


def write_openpi_config_patch(output_path: str | Path, **kwargs) -> dict[str, Any]:
    patch = build_openpi_config_patch(**kwargs)
    write_json(output_path, patch)
    return patch
