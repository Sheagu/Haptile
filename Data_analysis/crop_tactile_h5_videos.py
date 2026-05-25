#!/usr/bin/env python3
"""Select and apply task-specific crops for embedded tactile H5 videos.

Typical workflow:

1. Select four points on the earliest trajectory's tactile videos:

   python Data_analysis/crop_tactile_h5_videos.py select-config \
     shared/data/bc_data/put_bottle_upright \
     --config-dir sensor_configs/put_bottle_upright

2. Convert the full dataset into a new root:

   python Data_analysis/crop_tactile_h5_videos.py convert \
     shared/data/bc_data/put_bottle_upright \
     shared/data/bc_data/put_bottle_upright_tactile_crop \
     --config-dir sensor_configs/put_bottle_upright
     
按顺序点四个点：

左上, 右上, 右下, 左下
按键：

u = 撤销上一个点
r = 重选
Enter 或 s = 保存
q 或 Esc = 退出
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - clearer CLI failure
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc


TACTILE_VIDEO_KEYS = ("tactile_left_rgb", "tactile_right_rgb")
DEFAULT_OUTPUT_SIZE = (320, 240)  # width, height; matches existing DP tactile video size.


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select and apply perspective crops to /videos/tactile_*_rgb in trajectory.h5 files."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser(
        "select-config",
        help="Open the earliest H5 tactile frames and save four-point crop configs.",
    )
    select.add_argument("input_root", type=Path, help="Dataset root containing */trajectory.h5")
    select.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory to write tactile_left_rgb.json and tactile_right_rgb.json.",
    )
    select.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Optional episode directory name to use instead of the earliest sorted episode.",
    )
    select.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Video frame index used for point selection. Default: 0",
    )
    select.add_argument(
        "--output-size",
        type=parse_size,
        default=DEFAULT_OUTPUT_SIZE,
        help="Warp output size as WIDTHxHEIGHT. Default: 320x240",
    )
    select.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing config files.",
    )

    convert = subparsers.add_parser(
        "convert",
        help="Rewrite tactile embedded videos according to saved crop configs.",
    )
    convert.add_argument("input_root", type=Path, help="Input dataset root containing */trajectory.h5")
    convert.add_argument("output_root", type=Path, help="Output dataset root")
    convert.add_argument(
        "--config-dir",
        type=Path,
        required=True,
        help="Directory containing tactile_left_rgb.json and tactile_right_rgb.json.",
    )
    convert.add_argument(
        "--output-size",
        type=parse_size,
        default=None,
        help="Warp output size as WIDTHxHEIGHT. Defaults to config output_size, then 320x240.",
    )
    convert.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Convert only the first N trajectories.",
    )
    convert.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output episode directories.",
    )
    convert.add_argument(
        "--copy-extra-files",
        action="store_true",
        help="Copy files other than trajectory.h5 and freq.txt into each output episode.",
    )
    convert.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without writing files.",
    )

    preview = subparsers.add_parser(
        "preview",
        help="Write first-frame crop previews using saved configs.",
    )
    preview.add_argument("input_root", type=Path, help="Dataset root containing */trajectory.h5")
    preview.add_argument("--config-dir", type=Path, required=True)
    preview.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Preview image output directory. Default: <config-dir>/previews",
    )
    preview.add_argument("--episode", type=str, default=None)
    preview.add_argument("--frame-index", type=int, default=0)
    preview.add_argument("--output-size", type=parse_size, default=None)

    return parser.parse_args()


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError("Expected size formatted as WIDTHxHEIGHT, e.g. 320x240") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("Width and height must be positive")
    return width, height


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def adjusted_chunks(src: h5py.Dataset, shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if src.chunks is None or not shape:
        return None
    return tuple(max(1, min(chunk, dim)) for chunk, dim in zip(src.chunks, shape))


def create_dataset_like(dst_group: h5py.Group | h5py.File, name: str, src: h5py.Dataset, data: Any) -> h5py.Dataset:
    kwargs: dict[str, Any] = {"dtype": src.dtype}
    shape = np.shape(data)
    chunks = adjusted_chunks(src, shape)
    if chunks is not None:
        kwargs["chunks"] = chunks
    if src.compression is not None:
        kwargs["compression"] = src.compression
        kwargs["compression_opts"] = src.compression_opts
    if src.shuffle:
        kwargs["shuffle"] = src.shuffle
    if src.fletcher32:
        kwargs["fletcher32"] = src.fletcher32

    dst = dst_group.create_dataset(name, data=data, **kwargs)
    copy_attrs(src.attrs, dst.attrs)
    return dst


def find_h5_files(input_root: Path) -> list[Path]:
    return sorted(input_root.glob("*/trajectory.h5"))


def resolve_h5(input_root: Path, episode: str | None = None) -> Path:
    if episode is not None:
        h5_path = input_root / episode / "trajectory.h5"
        if not h5_path.is_file():
            raise FileNotFoundError(f"trajectory.h5 not found: {h5_path}")
        return h5_path

    h5_files = find_h5_files(input_root)
    if not h5_files:
        raise FileNotFoundError(f"No */trajectory.h5 files found under {input_root}")
    return h5_files[0]


def video_dataset_to_tempfile(dataset: h5py.Dataset) -> tempfile.NamedTemporaryFile:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(np.asarray(dataset, dtype=np.uint8).tobytes())
    tmp.close()
    return tmp


def read_video_frame(dataset: h5py.Dataset, frame_index: int) -> np.ndarray:
    tmp = video_dataset_to_tempfile(dataset)
    try:
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open embedded video: {dataset.name}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_index} from {dataset.name}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        Path(tmp.name).unlink(missing_ok=True)


def select_points(image_rgb: np.ndarray, window_name: str) -> list[list[int]]:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    display = image_bgr.copy()
    points: list[list[int]] = []

    def redraw() -> None:
        nonlocal display
        display = image_bgr.copy()
        for idx, (x, y) in enumerate(points):
            cv2.circle(display, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(
                display,
                str(idx + 1),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        if len(points) > 1:
            for a, b in zip(points, points[1:]):
                cv2.line(display, tuple(a), tuple(b), (255, 0, 0), 1)
        if len(points) == 4:
            cv2.line(display, tuple(points[-1]), tuple(points[0]), (255, 0, 0), 1)
        cv2.imshow(window_name, display)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([int(x), int(y)])
            print(f"{window_name}: point {len(points)} = [{x}, {y}]")
            redraw()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)
    print(
        f"\n{window_name}: click 4 points in order [top-left, top-right, bottom-right, bottom-left].\n"
        "Keys: u=undo, r=reset, Enter/s=save, q/Esc=quit"
    )
    redraw()
    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (13, ord("s")) and len(points) == 4:
            break
        if key == ord("u") and points:
            points.pop()
            redraw()
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key in (27, ord("q")):
            cv2.destroyWindow(window_name)
            raise RuntimeError(f"Point selection cancelled for {window_name}")
    cv2.destroyWindow(window_name)
    return points


def config_path(config_dir: Path, key: str) -> Path:
    return config_dir / f"{key}.json"


def write_config(path: Path, points: list[list[int]], output_size: tuple[int, int], source_h5: Path, frame_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "points": points,
        "output_size": list(output_size),
        "point_order": "top_left, top_right, bottom_right, bottom_left",
        "source_h5": str(source_h5),
        "source_frame_index": frame_index,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing crop config: {path}")
    config = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(config.get("points"), dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"{path} must contain 4 points shaped (4, 2), got {points.shape}")
    config["points"] = points
    return config


def warp_frame_bgr(frame_bgr: np.ndarray, points: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    dst_pts = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points.astype(np.float32), dst_pts)
    return cv2.warpPerspective(frame_bgr, transform, (width, height))


def crop_video_bytes(src: h5py.Dataset, config: dict[str, Any], output_size: tuple[int, int]) -> bytes:
    tmp = video_dataset_to_tempfile(src)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as dst_tmp:
        dst_path = Path(dst_tmp.name)

    try:
        cap = cv2.VideoCapture(tmp.name)
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open embedded video: {src.name}")

        fps = float(src.attrs.get("fps", cap.get(cv2.CAP_PROP_FPS) or 15.0))
        writer = cv2.VideoWriter(
            str(dst_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 15.0,
            output_size,
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"OpenCV could not create output video for {src.name}")

        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                writer.write(warp_frame_bgr(frame_bgr, config["points"], output_size))
        finally:
            writer.release()
            cap.release()

        return dst_path.read_bytes()
    finally:
        Path(tmp.name).unlink(missing_ok=True)
        dst_path.unlink(missing_ok=True)


def copy_group_with_cropped_tactile(
    src_group: h5py.Group,
    dst_group: h5py.Group,
    configs: dict[str, dict[str, Any]],
    output_size: tuple[int, int],
) -> None:
    copy_attrs(src_group.attrs, dst_group.attrs)
    for key, item in src_group.items():
        if isinstance(item, h5py.Group):
            child = dst_group.create_group(key)
            copy_group_with_cropped_tactile(item, child, configs, output_size)
            continue

        if not isinstance(item, h5py.Dataset):
            continue

        if item.name.startswith("/videos/") and key in configs:
            data = np.frombuffer(crop_video_bytes(item, configs[key], output_size), dtype=np.uint8)
            dst = dst_group.create_dataset(key, data=data, dtype=np.uint8, maxshape=(None,))
            copy_attrs(item.attrs, dst.attrs)
            dst.attrs["width"] = output_size[0]
            dst.attrs["height"] = output_size[1]
            dst.attrs["crop_points"] = configs[key]["points"]
            dst.attrs["crop_output_size"] = output_size
        else:
            create_dataset_like(dst_group, key, item, item[()])


def write_converted_h5(
    input_h5: Path,
    output_h5: Path,
    configs: dict[str, dict[str, Any]],
    output_size: tuple[int, int],
) -> None:
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(input_h5, "r") as src, h5py.File(output_h5, "w") as dst:
        copy_attrs(src.attrs, dst.attrs)
        for key, item in src.items():
            if isinstance(item, h5py.Group):
                child = dst.create_group(key)
                copy_group_with_cropped_tactile(item, child, configs, output_size)
            elif isinstance(item, h5py.Dataset):
                create_dataset_like(dst, key, item, item[()])


def command_select_config(args: argparse.Namespace) -> None:
    h5_path = resolve_h5(args.input_root, args.episode)
    print(f"Using source H5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        for key in TACTILE_VIDEO_KEYS:
            path = config_path(args.config_dir, key)
            if path.exists() and not args.overwrite:
                print(f"Skipping existing config {path} (use --overwrite to replace)")
                continue
            frame = read_video_frame(f[f"videos/{key}"], args.frame_index)
            points = select_points(frame, key)
            write_config(path, points, args.output_size, h5_path, args.frame_index)


def resolve_output_size(args_size: tuple[int, int] | None, configs: dict[str, dict[str, Any]]) -> tuple[int, int]:
    if args_size is not None:
        return args_size
    for config in configs.values():
        if "output_size" in config:
            return tuple(int(v) for v in config["output_size"])
    return DEFAULT_OUTPUT_SIZE


def load_configs(config_dir: Path) -> dict[str, dict[str, Any]]:
    return {key: load_config(config_path(config_dir, key)) for key in TACTILE_VIDEO_KEYS}


def copy_extra_files(input_dir: Path, output_dir: Path) -> None:
    for path in input_dir.iterdir():
        if path.name in {"trajectory.h5", "freq.txt"}:
            continue
        dst = output_dir / path.name
        if path.is_dir():
            shutil.copytree(path, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(path, dst)


def command_convert(args: argparse.Namespace) -> None:
    configs = load_configs(args.config_dir)
    output_size = resolve_output_size(args.output_size, configs)
    h5_files = find_h5_files(args.input_root)
    if args.limit is not None:
        h5_files = h5_files[: args.limit]
    if not h5_files:
        raise FileNotFoundError(f"No */trajectory.h5 files found under {args.input_root}")

    print(f"Input root: {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"Config dir: {args.config_dir}")
    print(f"Output tactile video size: {output_size[0]}x{output_size[1]}")
    print(f"Trajectories: {len(h5_files)}")

    for h5_path in h5_files:
        episode_name = h5_path.parent.name
        output_dir = args.output_root / episode_name
        output_h5 = output_dir / "trajectory.h5"
        if output_dir.exists():
            if args.overwrite:
                shutil.rmtree(output_dir)
            else:
                print(f"Skipping existing {output_dir} (use --overwrite to replace)")
                continue

        print(f"Converting {episode_name}")
        if args.dry_run:
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        write_converted_h5(h5_path, output_h5, configs, output_size)
        freq_path = h5_path.parent / "freq.txt"
        if freq_path.is_file():
            shutil.copy2(freq_path, output_dir / "freq.txt")
        if args.copy_extra_files:
            copy_extra_files(h5_path.parent, output_dir)


def command_preview(args: argparse.Namespace) -> None:
    configs = load_configs(args.config_dir)
    output_size = resolve_output_size(args.output_size, configs)
    h5_path = resolve_h5(args.input_root, args.episode)
    output_dir = args.output_dir or args.config_dir / "previews"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using source H5: {h5_path}")
    with h5py.File(h5_path, "r") as f:
        for key in TACTILE_VIDEO_KEYS:
            frame_rgb = read_video_frame(f[f"videos/{key}"], args.frame_index)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            cropped_bgr = warp_frame_bgr(frame_bgr, configs[key]["points"], output_size)
            out_path = output_dir / f"{key}_crop_preview.png"
            cv2.imwrite(str(out_path), cropped_bgr)
            print(f"Wrote {out_path}")


def main() -> None:
    args = parse_args()
    if args.command == "select-config":
        command_select_config(args)
    elif args.command == "convert":
        command_convert(args)
    elif args.command == "preview":
        command_preview(args)
    else:  # pragma: no cover
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
