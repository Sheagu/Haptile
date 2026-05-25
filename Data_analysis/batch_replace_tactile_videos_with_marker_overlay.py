#!/usr/bin/env python3
"""Replace embedded tactile RGB videos with marker-tracking overlay videos.

用途：
    批量读取 DATA_ROOT/*/trajectory.h5 中的：

        /videos/tactile_left_rgb
        /videos/tactile_right_rgb

    然后参考采集时的 marker tracking 方法，生成带绿色 marker 点和红色
    flow 箭头的 overlay 视频，并直接替换回同名 H5 video dataset。

重要：
    这个脚本会原地修改输入数据集里的 trajectory.h5。建议只对已经备份过
    或另存出来的数据集运行，例如 *_tactile_crop。

常用命令：
    # 1. 先 dry-run，确认会处理哪些 h5 和 stream，不写文件
    conda run -n tele python Data_analysis/batch_replace_tactile_videos_with_marker_overlay.py \
      shared/data/bc_data/put_bottle_upright_tactile_crop \
      --dry-run

    # 2. 批量替换整个数据集的左右 tactile RGB 视频
    conda run -n tele python Data_analysis/batch_replace_tactile_videos_with_marker_overlay.py \
      shared/data/bc_data/put_bottle_upright_tactile_crop

    # 3. 只测试前 N 条轨迹
    conda run -n tele python Data_analysis/batch_replace_tactile_videos_with_marker_overlay.py \
      shared/data/bc_data/put_bottle_upright_tactile_crop \
      --limit 3

    # 4. 只替换右手触觉视频
    conda run -n tele python Data_analysis/batch_replace_tactile_videos_with_marker_overlay.py \
      shared/data/bc_data/put_bottle_upright_tactile_crop \
      --streams tactile_right_rgb

    # 5. 如果已经替换过，默认会跳过；如需重新生成，显式加 --overwrite-overlay
    conda run -n tele python Data_analysis/batch_replace_tactile_videos_with_marker_overlay.py \
      shared/data/bc_data/put_bottle_upright_tactile_crop \
      --overwrite-overlay

输出：
    替换后的 H5 dataset 仍然叫：

        /videos/tactile_left_rgb
        /videos/tactile_right_rgb

    并会在 attrs 中写入：

        marker_tracking_overlay = True
        marker_tracking_overlay_generator = "Data_analysis/export_marker_tracking_overlay.py"
        marker_tracking_overlay_frames = ...
        marker_tracking_overlay_motion_mean = ...
        marker_tracking_overlay_motion_max = ...

    这些标记用于避免重复运行时把箭头再次画到已经 overlay 的视频上。
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import h5py
import numpy as np

from export_marker_tracking_overlay import TACTILE_VIDEO_KEYS, export_overlay_video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch replace /videos/tactile_left_rgb and /videos/tactile_right_rgb "
            "with marker-tracking arrow overlay videos."
        )
    )
    parser.add_argument("data_root", type=Path, help="Dataset root containing */trajectory.h5")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N trajectories")
    parser.add_argument("--dry-run", action="store_true", help="Print planned work without writing h5 files")
    parser.add_argument(
        "--overwrite-overlay",
        action="store_true",
        help="Regenerate streams already tagged as marker_tracking_overlay.",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=list(TACTILE_VIDEO_KEYS),
        choices=list(TACTILE_VIDEO_KEYS),
        help="Streams to replace.",
    )

    # Keep these defaults aligned with export_marker_tracking_overlay.py.
    parser.add_argument("--arrow-scale", type=float, default=6.0)
    parser.add_argument("--flow-win-size", type=int, nargs=2, default=(15, 15))
    parser.add_argument("--flow-max-level", type=int, default=2)
    parser.add_argument("--flow-fb-max-error", type=float, default=1.5)
    parser.add_argument("--require-marker-mask", type=str_to_bool, default=False)
    parser.add_argument("--mask-point-radius", type=int, default=3)
    parser.add_argument("--motion-deadband", type=float, default=0.2)
    parser.add_argument("--motion-smoothing", type=float, default=0.2)
    parser.add_argument("--motion-release-smoothing", type=float, default=0.8)
    parser.add_argument("--min-valid-points", type=int, default=8)
    parser.add_argument("--compensate-global-drift", type=str_to_bool, default=True)
    parser.add_argument("--reset-on-loss", type=str_to_bool, default=True)
    parser.add_argument("--marker-mask-range", type=int, nargs=2, default=(145, 255))
    parser.add_argument("--marker-value-threshold", type=int, default=90)
    parser.add_argument("--marker-morph-open-size", type=int, default=5)
    parser.add_argument("--marker-morph-open-iter", type=int, default=1)
    parser.add_argument("--marker-morph-close-size", type=int, default=5)
    parser.add_argument("--marker-morph-close-iter", type=int, default=1)
    parser.add_argument("--marker-dilate-size", type=int, default=3)
    parser.add_argument("--marker-dilate-iter", type=int, default=0)
    return parser.parse_args()


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def find_h5_files(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("*/trajectory.h5"))


def replace_video_dataset(
    videos_group: h5py.Group,
    key: str,
    video_bytes: bytes,
    info: dict,
) -> None:
    old_ds = videos_group[key]
    attrs = dict(old_ds.attrs)
    del videos_group[key]

    new_ds = videos_group.create_dataset(
        key,
        data=np.frombuffer(video_bytes, dtype=np.uint8),
        dtype=np.uint8,
        maxshape=(None,),
    )
    for attr_key, attr_value in attrs.items():
        new_ds.attrs[attr_key] = attr_value
    new_ds.attrs["marker_tracking_overlay"] = True
    new_ds.attrs["marker_tracking_overlay_source"] = key
    new_ds.attrs["marker_tracking_overlay_generator"] = "Data_analysis/export_marker_tracking_overlay.py"
    new_ds.attrs["marker_tracking_overlay_frames"] = int(info["frames"])
    new_ds.attrs["marker_tracking_overlay_motion_mean"] = float(info["motion_mean"])
    new_ds.attrs["marker_tracking_overlay_motion_max"] = float(info["motion_max"])
    new_ds.attrs["width"] = int(info["width"])
    new_ds.attrs["height"] = int(info["height"])
    new_ds.attrs["fps"] = float(info["fps"])


def overlay_bytes_for_dataset(dataset: h5py.Dataset, sensor_name: str, args: argparse.Namespace) -> tuple[bytes, dict]:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        output_path = Path(tmp.name)
    try:
        info = export_overlay_video(dataset, output_path, sensor_name, args)
        return output_path.read_bytes(), info
    finally:
        output_path.unlink(missing_ok=True)


def process_h5(h5_path: Path, args: argparse.Namespace) -> tuple[int, int]:
    replaced = 0
    skipped = 0
    print(f"Processing {h5_path}")
    if args.dry_run:
        with h5py.File(h5_path, "r") as f:
            for key in args.streams:
                if key not in f["videos"]:
                    print(f"  missing /videos/{key}")
                    skipped += 1
                    continue
                already = bool(f["videos"][key].attrs.get("marker_tracking_overlay", False))
                action = "would replace"
                if already and not args.overwrite_overlay:
                    action = "would skip existing overlay"
                    skipped += 1
                else:
                    replaced += 1
                print(f"  {action} /videos/{key}")
        return replaced, skipped

    with h5py.File(h5_path, "r+") as f:
        if "videos" not in f:
            print("  missing /videos group")
            return replaced, len(args.streams)

        generated: dict[str, tuple[bytes, dict]] = {}
        for key in args.streams:
            if key not in f["videos"]:
                print(f"  missing /videos/{key}")
                skipped += 1
                continue
            ds = f["videos"][key]
            already = bool(ds.attrs.get("marker_tracking_overlay", False))
            if already and not args.overwrite_overlay:
                print(f"  skip /videos/{key}: already marker_tracking_overlay")
                skipped += 1
                continue
            sensor_name = key.removesuffix("_rgb")
            video_bytes, info = overlay_bytes_for_dataset(ds, sensor_name, args)
            generated[key] = (video_bytes, info)

        for key, (video_bytes, info) in generated.items():
            replace_video_dataset(f["videos"], key, video_bytes, info)
            print(
                f"  replaced /videos/{key}: "
                f"frames={info['frames']} fps={info['fps']:.2f} "
                f"size={info['width']}x{info['height']}"
            )
            replaced += 1
        f.flush()

    return replaced, skipped


def main() -> None:
    args = parse_args()
    h5_files = find_h5_files(args.data_root)
    if args.limit is not None:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        raise FileNotFoundError(f"No */trajectory.h5 files found under {args.data_root}")

    print(f"Data root: {args.data_root}")
    print(f"Trajectories: {len(h5_files)}")
    print(f"Streams: {', '.join(args.streams)}")
    print(f"Dry run: {args.dry_run}")

    total_replaced = 0
    total_skipped = 0
    for h5_path in h5_files:
        replaced, skipped = process_h5(h5_path, args)
        total_replaced += replaced
        total_skipped += skipped

    print(f"Done. replaced={total_replaced}, skipped={total_skipped}")


if __name__ == "__main__":
    main()
