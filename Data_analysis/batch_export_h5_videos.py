#!/usr/bin/env python3
"""Batch export videos from trajectory.h5 files using test_h5_video_export.py.

用法:
    # 在项目根目录运行；DATA_ROOT 是包含多个数据集子目录的目录。
    # 脚本会寻找 DATA_ROOT/*/trajectory.h5，并为每个 h5 导出视频。
    python Data_analysis/batch_export_h5_videos.py DATA_ROOT

示例:
    python Data_analysis/batch_export_h5_videos.py shared/data/bc_data/grab_05

常用选项:
    # 只先导出前 3 个数据集做测试
    python Data_analysis/batch_export_h5_videos.py shared/data/bc_data/grab_05 --limit 3

    # 指定输出目录，默认是 DATA_ROOT/exported_videos_batch
    python Data_analysis/batch_export_h5_videos.py shared/data/bc_data/grab_05 \
        --output-root shared/data/bc_data/grab_05/exported_videos_batch

    # 已经导出过 combined_2x2.mp4 时默认会跳过；加 --overwrite 可重新导出
    python Data_analysis/batch_export_h5_videos.py shared/data/bc_data/grab_05 --overwrite

    # 只汇总导出 combined_2x2 视频；输出为同一层的 <原文件夹时间戳>.mp4
    python Data_analysis/batch_export_h5_videos.py shared/data/bc_data/grab_05 --flat-combined-only

输出:
    # 默认模式
    DATA_ROOT/exported_videos_batch/<dataset_name>/
        base_camera_rgb_0.mp4
        base_camera_rgb_1.mp4
        tactile_left_rgb.mp4
        tactile_right_rgb.mp4
        combined_2x2.mp4
        ... depth 视频及 depth 可视化视频（如果 h5 中包含）

    # --flat-combined-only 模式
    DATA_ROOT/exported_videos_batch/<dataset_name>.mp4
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
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
    parser.add_argument(
        "--flat-combined-only",
        action="store_true",
        help=(
            "Only keep the combined_2x2 video for each dataset, written as "
            "<dataset_folder_name>.mp4 directly under output root."
        ),
    )
    return parser.parse_args()


def export_flat_combined_video(
    *,
    python_executable: str,
    script_path: Path,
    h5_path: Path,
    output_path: Path,
) -> None:
    with tempfile.TemporaryDirectory(
        prefix=f".{output_path.stem}_",
        dir=output_path.parent,
    ) as tmp:
        tmp_output_dir = Path(tmp)
        cmd = [
            python_executable,
            str(script_path),
            str(h5_path),
            "--output-dir",
            str(tmp_output_dir),
        ]
        subprocess.run(cmd, check=True)

        combined_path = tmp_output_dir / "combined_2x2.mp4"
        if not combined_path.is_file():
            raise FileNotFoundError(f"Combined video was not generated: {combined_path}")
        shutil.copy2(combined_path, output_path)


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
    print(f"Mode:        {'flat combined only' if args.flat_combined_only else 'full export'}")

    success = 0
    failed = 0
    skipped = 0
    start = time.time()

    for idx, dataset_dir in enumerate(dataset_dirs, start=1):
        dataset_name = dataset_dir.name
        h5_path = dataset_dir / "trajectory.h5"
        if args.flat_combined_only:
            output_path = output_root / f"{dataset_name}.mp4"
            done_marker = output_path
        else:
            dataset_output_dir = output_root / dataset_name
            done_marker = dataset_output_dir / "combined_2x2.mp4"

        print(f"\n[{idx}/{len(dataset_dirs)}] {dataset_name}")
        print(f"  H5:      {h5_path}")
        if args.flat_combined_only:
            print(f"  Output:  {output_path}")
        else:
            print(f"  Output:  {dataset_output_dir}")

        if done_marker.exists() and not args.overwrite:
            print(f"  Skipped: {done_marker.name} already exists")
            skipped += 1
            continue

        try:
            if args.flat_combined_only:
                export_flat_combined_video(
                    python_executable=args.python,
                    script_path=script_path,
                    h5_path=h5_path,
                    output_path=output_path,
                )
            else:
                cmd = [
                    args.python,
                    str(script_path),
                    str(h5_path),
                    "--output-dir",
                    str(dataset_output_dir),
                ]
                subprocess.run(cmd, check=True)
            print("  Success")
            success += 1
        except subprocess.CalledProcessError as exc:
            print(f"  Failed with exit code {exc.returncode}")
            failed += 1
        except Exception as exc:
            print(f"  Failed: {exc}")
            failed += 1

    duration = time.time() - start
    print("\nSummary")
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed:  {failed}")
    print(f"  Time:    {duration:.1f}s")


if __name__ == "__main__":
    main()
