#!/usr/bin/env python3
"""Summarize TeleUR dataset folders."""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count trajectories, total frames, and durations in a dataset folder."
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory containing trajectory subfolders.",
    )
    return parser.parse_args()


def parse_legacy_timestamp(path: Path) -> dt.datetime:
    return dt.datetime.strptime(path.stem, "%Y-%m-%dT%H-%M-%S-%f")


def stats_from_h5(h5_path: Path) -> tuple[int, float]:
    with h5py.File(h5_path, "r") as f:
        frame_count = int(f.attrs.get("frame_count", 0))
        if "timestamps" not in f or len(f["timestamps"]) < 2:
            fps = float(f.attrs.get("video_fps", 0.0))
            duration = (frame_count / fps) if fps > 0 else 0.0
            return frame_count, duration

        timestamps = [
            dt.datetime.fromisoformat(x.decode() if isinstance(x, bytes) else str(x))
            for x in f["timestamps"][()]
        ]
        duration = max(0.0, (timestamps[-1] - timestamps[0]).total_seconds())
        return frame_count, duration


def stats_from_pkl_dir(dataset_dir: Path) -> tuple[int, float]:
    pkl_paths = sorted(dataset_dir.glob("*.pkl"))
    frame_count = len(pkl_paths)
    if frame_count < 2:
        return frame_count, 0.0
    timestamps = [parse_legacy_timestamp(path) for path in pkl_paths]
    duration = max(0.0, (timestamps[-1] - timestamps[0]).total_seconds())
    return frame_count, duration


def collect_dataset_stats(dataset_dir: Path) -> tuple[int, float] | None:
    h5_path = dataset_dir / "trajectory.h5"
    if h5_path.is_file():
        return stats_from_h5(h5_path)

    pkl_paths = list(dataset_dir.glob("*.pkl"))
    if pkl_paths:
        return stats_from_pkl_dir(dataset_dir)

    return None


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    dataset_dirs = [path for path in sorted(data_root.iterdir()) if path.is_dir()]

    num_datasets = 0
    total_frames = 0
    total_duration_s = 0.0

    for dataset_dir in dataset_dirs:
        stats = collect_dataset_stats(dataset_dir)
        if stats is None:
            continue
        frames, duration_s = stats
        num_datasets += 1
        total_frames += frames
        total_duration_s += duration_s

    avg_frames = (total_frames / num_datasets) if num_datasets > 0 else 0.0
    avg_duration_s = (total_duration_s / num_datasets) if num_datasets > 0 else 0.0

    print(f"Data root: {data_root}")
    print(f"Datasets: {num_datasets}")
    print(f"Total frames: {total_frames}")
    print(f"Total duration (min): {total_duration_s / 60.0:.2f}")
    print(f"Average frames per dataset: {avg_frames:.2f}")
    print(f"Average duration per dataset (s): {avg_duration_s:.2f}")


if __name__ == "__main__":
    main()
