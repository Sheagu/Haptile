#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi0_ur5e.dataset_reader import DatasetReader
from pi0_ur5e.io_utils import write_json


def stats(x):
    return {"mean": np.mean(x, axis=0).tolist(), "std": np.std(x, axis=0).tolist(), "min": np.min(x, axis=0).tolist(), "max": np.max(x, axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser(description="Compute state/action normalization stats.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", default=Path("learning/pi0_ur5e/configs/dataset_schema.yaml"), type=Path)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()
    episodes = DatasetReader(args.dataset_root, args.config).episodes()
    states = np.concatenate([ep.robot_state.astype(np.float32) for ep in episodes], axis=0)
    actions = np.concatenate([ep.action.astype(np.float32) for ep in episodes], axis=0)
    result = {"state": stats(states), "action": stats(actions)}
    output = args.output or (args.dataset_root / "norm_stats.json")
    write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
