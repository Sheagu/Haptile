#!/usr/bin/env python3
"""Export marker-tracking overlay videos from embedded tactile RGB H5 videos.

This mirrors the online marker tracking visualization used during collection:
marker detection on the reference frame, Lucas-Kanade optical flow from the
reference frame to each current frame, optional global-drift compensation, then
green marker points and red displacement arrows drawn on the tactile image.

python Data_analysis/export_marker_tracking_overlay.py \
  shared/data/bc_data/put_bottle_upright_tactile_crop/0518_170045/trajectory.h5
  
"""

from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - CLI dependency error
    raise RuntimeError("OpenCV is required: pip install opencv-python") from exc

from marker_tracking.utils import find_marker, find_marker_centers, plot_marker_delta


TACTILE_VIDEO_KEYS = ("tactile_left_rgb", "tactile_right_rgb")


@dataclass
class MarkerTrackingState:
    name: str
    ref_gray: np.ndarray | None = None
    ref_points: np.ndarray | None = None
    motion_ema: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate marker-tracking arrow overlay MP4s from /videos/tactile_*_rgb in a trajectory.h5."
    )
    parser.add_argument("h5_path", type=Path, help="Path to trajectory.h5")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for exported MP4s. Default: same directory as h5_path.",
    )
    parser.add_argument(
        "--output-prefix",
        default="",
        help="Optional filename prefix for exported videos.",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=list(TACTILE_VIDEO_KEYS),
        choices=list(TACTILE_VIDEO_KEYS),
        help="Tactile streams to process.",
    )
    parser.add_argument("--arrow-scale", type=float, default=6.0)
    parser.add_argument("--flow-win-size", type=int, nargs=2, default=(15, 15))
    parser.add_argument("--flow-max-level", type=int, default=2)
    parser.add_argument("--flow-fb-max-error", type=float, default=1.5)
    parser.add_argument(
        "--require-marker-mask",
        type=str_to_bool,
        default=False,
        help=(
            "Require tracked points to stay on the marker segmentation mask. "
            "Default false for offline visualization so weak edge markers that "
            "track well by LK are still drawn."
        ),
    )
    parser.add_argument(
        "--mask-point-radius",
        type=int,
        default=3,
        help=(
            "Accept a tracked point if any marker-mask pixel exists within this "
            "radius. This reduces flicker from tiny mask gaps. Default: 3"
        ),
    )
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


def build_marker_tracking_params(args: argparse.Namespace) -> dict:
    return {
        "morphop_kernel": cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.marker_morph_open_size, args.marker_morph_open_size),
        ),
        "morphclose_kernel": cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.marker_morph_close_size, args.marker_morph_close_size),
        ),
        "dilate_kernel": cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (args.marker_dilate_size, args.marker_dilate_size),
        ),
        "mask_range": tuple(args.marker_mask_range),
        "min_value": args.marker_value_threshold,
        "morphop_iter": args.marker_morph_open_iter,
        "morphclose_iter": args.marker_morph_close_iter,
        "dilate_iter": args.marker_dilate_iter,
    }


def init_marker_tracking_state(
    sensor_name: str,
    frame_rgb: np.ndarray,
    marker_tracking_params: dict,
) -> MarkerTrackingState:
    marker_mask = find_marker(frame_rgb, **marker_tracking_params)
    centers = find_marker_centers(marker_mask)
    if not centers:
        print(f"[marker_tracking] {sensor_name}: no markers found in reference frame")
        return MarkerTrackingState(name=sensor_name)

    state = MarkerTrackingState(
        name=sensor_name,
        ref_gray=cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY),
        ref_points=np.array(centers, dtype=np.float32).reshape(-1, 1, 2),
    )
    print(f"[marker_tracking] {sensor_name}: initialized with {len(centers)} markers")
    return state


def mask_contains_points(mask: np.ndarray, points: np.ndarray, radius: int = 3) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(0, dtype=bool)

    if radius > 0:
        kernel_size = radius * 2 + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        mask = cv2.dilate(mask, kernel, iterations=1)

    h, w = mask.shape[:2]
    rounded = np.rint(points).astype(np.int32)
    in_bounds = (
        (rounded[:, 0] >= 0)
        & (rounded[:, 0] < w)
        & (rounded[:, 1] >= 0)
        & (rounded[:, 1] < h)
    )
    on_mask = np.zeros(len(points), dtype=bool)
    if np.any(in_bounds):
        valid_points = rounded[in_bounds]
        on_mask[in_bounds] = mask[valid_points[:, 1], valid_points[:, 0]] > 0
    return on_mask


def update_marker_tracking(
    sensor_name: str,
    frame_rgb: np.ndarray,
    state: MarkerTrackingState,
    marker_tracking_params: dict,
    lk_params: dict,
    *,
    reset_on_loss: bool,
    arrow_scale: float,
    fb_max_error: float,
    motion_deadband: float,
    motion_smoothing: float,
    motion_release_smoothing: float,
    min_valid_points: int,
    compensate_global_drift: bool,
    mask_point_radius: int,
    require_marker_mask: bool,
) -> tuple[MarkerTrackingState, np.ndarray, float]:
    if state.ref_points is None or state.ref_gray is None:
        state = init_marker_tracking_state(sensor_name, frame_rgb, marker_tracking_params)
        return state, frame_rgb, 0.0

    track_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        state.ref_gray,
        track_gray,
        state.ref_points,
        None,
        **lk_params,
    )
    if next_points is None or status is None:
        if reset_on_loss:
            state = init_marker_tracking_state(sensor_name, frame_rgb, marker_tracking_params)
        return state, frame_rgb, 0.0

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        track_gray,
        state.ref_gray,
        next_points,
        None,
        **lk_params,
    )
    if back_points is None or back_status is None:
        if reset_on_loss:
            state = init_marker_tracking_state(sensor_name, frame_rgb, marker_tracking_params)
        return state, frame_rgb, 0.0

    current_marker_mask = find_marker(frame_rgb, **marker_tracking_params)
    ref_points_all = state.ref_points.reshape(-1, 2)
    tracked_points_all = next_points.reshape(-1, 2)
    back_points_all = back_points.reshape(-1, 2)
    status = status.reshape(-1).astype(bool)
    back_status = back_status.reshape(-1).astype(bool)
    fb_error = np.linalg.norm(back_points_all - ref_points_all, axis=1)
    valid = status & back_status & (fb_error <= fb_max_error)
    if require_marker_mask:
        on_marker_mask = mask_contains_points(
            current_marker_mask, tracked_points_all, radius=mask_point_radius
        )
        valid = valid & on_marker_mask

    if np.count_nonzero(valid) < min_valid_points:
        if reset_on_loss:
            state = init_marker_tracking_state(sensor_name, frame_rgb, marker_tracking_params)
        return state, frame_rgb, 0.0

    ref_points = ref_points_all[valid]
    tracked_points = tracked_points_all[valid]
    deltas = tracked_points - ref_points
    if compensate_global_drift and len(deltas) > 0:
        deltas = deltas - np.median(deltas, axis=0, keepdims=True)

    delta_norms = np.linalg.norm(deltas, axis=1)
    delta_norms = np.clip(delta_norms - motion_deadband, 0.0, None)
    sum_motion = float(delta_norms.sum()) if len(delta_norms) > 0 else 0.0
    alpha = motion_release_smoothing if sum_motion < state.motion_ema else motion_smoothing
    alpha = float(np.clip(alpha, 0.0, 1.0))
    state.motion_ema = (1.0 - alpha) * state.motion_ema + alpha * sum_motion
    overlay = plot_marker_delta(
        frame_rgb,
        tracked_points,
        deltas,
        scale=arrow_scale,
        arrow_color=(255, 0, 0),
    )
    return state, overlay, state.motion_ema


def write_dataset_to_temp_mp4(dataset: h5py.Dataset) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(np.asarray(dataset, dtype=np.uint8).tobytes())
    tmp.close()
    return Path(tmp.name)


def export_overlay_video(
    dataset: h5py.Dataset,
    output_path: Path,
    sensor_name: str,
    args: argparse.Namespace,
) -> dict:
    marker_tracking_params = build_marker_tracking_params(args)
    lk_params = {
        "winSize": tuple(args.flow_win_size),
        "maxLevel": args.flow_max_level,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
    }
    tmp_path = write_dataset_to_temp_mp4(dataset)
    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV could not open embedded video: {dataset.name}")

        fps = float(dataset.attrs.get("fps", cap.get(cv2.CAP_PROP_FPS) or 15.0))
        width = int(dataset.attrs.get("width", cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(dataset.attrs.get("height", cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps if fps > 0 else 15.0,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"OpenCV could not create output video: {output_path}")

        state: MarkerTrackingState | None = None
        frame_count = 0
        motion_values: list[float] = []
        try:
            while True:
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                if frame_bgr.shape[1] != width or frame_bgr.shape[0] != height:
                    frame_bgr = cv2.resize(frame_bgr, (width, height), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                if state is None:
                    state = init_marker_tracking_state(
                        sensor_name, frame_rgb, marker_tracking_params
                    )
                state, overlay_rgb, motion = update_marker_tracking(
                    sensor_name,
                    frame_rgb,
                    state,
                    marker_tracking_params,
                    lk_params,
                    reset_on_loss=args.reset_on_loss,
                    arrow_scale=args.arrow_scale,
                    fb_max_error=args.flow_fb_max_error,
                    motion_deadband=args.motion_deadband,
                    motion_smoothing=args.motion_smoothing,
                    motion_release_smoothing=args.motion_release_smoothing,
                    min_valid_points=args.min_valid_points,
                    compensate_global_drift=args.compensate_global_drift,
                    mask_point_radius=args.mask_point_radius,
                    require_marker_mask=args.require_marker_mask,
                )
                writer.write(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))
                motion_values.append(float(motion))
                frame_count += 1
        finally:
            writer.release()
            cap.release()

        return {
            "frames": frame_count,
            "fps": fps,
            "width": width,
            "height": height,
            "motion_min": min(motion_values) if motion_values else 0.0,
            "motion_max": max(motion_values) if motion_values else 0.0,
            "motion_mean": float(np.mean(motion_values)) if motion_values else 0.0,
        }
    finally:
        tmp_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.h5_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.h5_path, "r") as f:
        if "videos" not in f:
            raise KeyError(f"No /videos group found in {args.h5_path}")
        for key in args.streams:
            if key not in f["videos"]:
                raise KeyError(f"Missing /videos/{key} in {args.h5_path}")
            sensor_name = key.removesuffix("_rgb")
            output_path = output_dir / f"{args.output_prefix}{sensor_name}_marker_tracking.mp4"
            info = export_overlay_video(f["videos"][key], output_path, sensor_name, args)
            print(
                f"Wrote {output_path} "
                f"frames={info['frames']} fps={info['fps']:.2f} "
                f"size={info['width']}x{info['height']} "
                f"motion_mean={info['motion_mean']:.4f} motion_max={info['motion_max']:.4f}"
            )


if __name__ == "__main__":
    main()
