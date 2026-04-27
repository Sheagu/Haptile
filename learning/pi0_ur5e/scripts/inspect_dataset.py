#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi0_ur5e.dataset_reader import DatasetReader
from pi0_ur5e.sanity_checks import inspect_episodes


def main():
    parser = argparse.ArgumentParser(description="Inspect raw or converted UR5e pi0 dataset.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", default=Path("learning/pi0_ur5e/configs/dataset_schema.yaml"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--include-tactile", default="false")
    args = parser.parse_args()
    include_tactile = str(args.include_tactile).lower() == "true"
    episodes = DatasetReader(args.dataset_root, args.config, config={"include_tactile": include_tactile}).episodes()
    if not episodes:
        raise SystemExit(f"No readable episodes found under {args.dataset_root}")
    report = inspect_episodes(episodes, args.output_dir, include_tactile=include_tactile)
    print(f"Inspected {report['episode_count']} episodes / {report['total_frames']} frames")
    print(f"Report: {args.output_dir / 'inspection_report.json'}")
    print(f"Debug images: {args.output_dir / 'debug_vis'}")


if __name__ == "__main__":
    main()
