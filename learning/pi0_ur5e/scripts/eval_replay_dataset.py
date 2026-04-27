#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pi0_ur5e.dataset_reader import DatasetReader
from pi0_ur5e.visualization import plot_replay


def main():
    parser = argparse.ArgumentParser(description="Replay/evaluate dataset actions without controlling the real robot by default.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--config", default=Path("learning/pi0_ur5e/configs/dataset_schema.yaml"), type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--allow-real-robot", default="false")
    args = parser.parse_args()
    if str(args.allow_real_robot).lower() == "true":
        raise SystemExit("Real robot replay adapter is not implemented here; keep this script offline or add a project-specific safe adapter.")
    episodes = DatasetReader(args.dataset_root, args.config).episodes()[: args.num_episodes]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for episode in episodes:
        plot_replay(episode, args.output_dir / f"{episode.episode_id}_replay.png")
    print(f"Wrote replay plots to {args.output_dir}")


if __name__ == "__main__":
    main()
