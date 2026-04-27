#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi0_ur5e.dataset_reader import DatasetReader
from pi0_ur5e.lerobot_writer import write_lerobot_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Convert UR5e raw trajectories to a LeRobot/OpenPI-style dataset.")
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--config", default=Path("learning/pi0_ur5e/configs/dataset_schema.yaml"), type=Path)
    parser.add_argument("--task-name", default="cup_pick_place")
    parser.add_argument("--default-prompt", default="pick up the paper cup and place it on the target")
    parser.add_argument("--include-tactile", default="false")
    parser.add_argument("--repo-id", default="local/pi0_ur5e_cup")
    parser.add_argument("--overwrite", default="false")
    return parser.parse_args()


def main():
    args = parse_args()
    include_tactile = str(args.include_tactile).lower() == "true"
    reader = DatasetReader(args.input_root, args.config, config={"include_tactile": include_tactile, "default_prompt": args.default_prompt})
    episodes = reader.episodes()
    if not episodes:
        raise SystemExit(f"No readable episodes found under {args.input_root}")
    report = write_lerobot_dataset(
        episodes,
        args.output_root,
        task_name=args.task_name,
        image_size=reader.config.image_size,
        include_tactile=include_tactile,
        camera_padding_strategy=reader.config.camera_padding_strategy,
        repo_id=args.repo_id,
        overwrite=str(args.overwrite).lower() == "true",
    )
    print(f"Converted {report['episode_count']} episodes / {report['total_frames']} frames to {args.output_root}")
    print(f"Report: {args.output_root / 'conversion_report.json'}")


if __name__ == "__main__":
    main()
