#!/usr/bin/env python3
"""Batch export videos from trajectory.h5 files using test_h5_video_export.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch export embedded videos from TeleUR trajectory.h5 datasets."
    )
    parser.add_argument(
        "data_root",
        type=Path,
        help="Root directory containing dataset subfolders, e.g. shared/data/bc_data/grab_05",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for exported videos. Defaults to <data_root>/exported_videos_batch",
    )
    parser.add_argument(
        "--script",
        type=Path,
        default=Path(__file__).resolve().parent / "test_h5_video_export.py",
        help="Path to test_h5_video_export.py",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use. Example: /home/shiyigu/anaconda3/envs/tele/bin/python",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Export only the first N datasets for testing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-export even if output directory already contains combined_2x2.mp4.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    script_path = args.script.expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not script_path.is_file():
        raise FileNotFoundError(f"Export script not found: {script_path}")

    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else data_root / "exported_videos_batch"
    )
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_dirs = []
    for item in sorted(data_root.iterdir()):
        if not item.is_dir():
            continue
        h5_path = item / "trajectory.h5"
        if h5_path.is_file():
            dataset_dirs.append(item)

    if args.limit is not None:
        dataset_dirs = dataset_dirs[: args.limit]

    print(f"Data root:   {data_root}")
    print(f"Output root: {output_root}")
    print(f"Datasets:    {len(dataset_dirs)}")
    print(f"Exporter:    {script_path}")
    print(f"Python:      {args.python}")

    success = 0
    failed = 0
    skipped = 0
    start = time.time()

    for idx, dataset_dir in enumerate(dataset_dirs, start=1):
        dataset_name = dataset_dir.name
        h5_path = dataset_dir / "trajectory.h5"
        dataset_output_dir = output_root / dataset_name
        done_marker = dataset_output_dir / "combined_2x2.mp4"

        print(f"\n[{idx}/{len(dataset_dirs)}] {dataset_name}")
        print(f"  H5:      {h5_path}")
        print(f"  Output:  {dataset_output_dir}")

        if done_marker.exists() and not args.overwrite:
            print("  Skipped: combined_2x2.mp4 already exists")
            skipped += 1
            continue

        cmd = [
            args.python,
            str(script_path),
            str(h5_path),
            "--output-dir",
            str(dataset_output_dir),
        ]

        try:
            subprocess.run(cmd, check=True)
            print("  Success")
            success += 1
        except subprocess.CalledProcessError as exc:
            print(f"  Failed with exit code {exc.returncode}")
            failed += 1

    duration = time.time() - start
    print("\nSummary")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Time:    {duration:.1f}s")


if __name__ == "__main__":
    main()
