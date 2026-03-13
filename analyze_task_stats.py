#!/usr/bin/env python3
"""
Analyze task-level dataset statistics for TeleUR recordings.

For each dataset root (one task), this script reports:
- number of demos
- total number of frames
- total duration in minutes

Each demo is assumed to be a subdirectory containing `.pkl` frame files.
Duration is estimated in this order:
1. Timestamp span from first/last `.pkl` filename
2. Average FPS from `freq.txt`
3. A default FPS provided by CLI
"""

from __future__ import annotations

import argparse
import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASETS = [
    str(PROJECT_ROOT / "shared/data/huawei_grab_03"),
    str(PROJECT_ROOT / "shared/data/huawei_grab_05"),
    str(PROJECT_ROOT / "shared/data/huawei_grab_23"),
]

SKIP_SUFFIXES = ("failed", "ood", "ikbad", "heated", "stop", "hard")
TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}(?:-\d+)?$"
)
FPS_RE = re.compile(r"Average FPS:\s*([0-9]+(?:\.[0-9]+)?)")


@dataclass
class DemoStats:
    demo_dir: Path
    frames: int
    seconds: float
    duration_source: str


@dataclass
class TaskStats:
    task_dir: Path
    demos: list[DemoStats]

    @property
    def num_demos(self) -> int:
        return len(self.demos)

    @property
    def total_frames(self) -> int:
        return sum(demo.frames for demo in self.demos)

    @property
    def total_seconds(self) -> float:
        return sum(demo.seconds for demo in self.demos)

    @property
    def avg_frames(self) -> float:
        if not self.demos:
            return 0.0
        return self.total_frames / self.num_demos

    @property
    def avg_seconds(self) -> float:
        if not self.demos:
            return 0.0
        return self.total_seconds / self.num_demos


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze num demos, minutes, and frames for task datasets."
    )
    parser.add_argument(
        "dataset_paths",
        nargs="*",
        default=DEFAULT_DATASETS,
        help="Task dataset roots. Defaults to the three Huawei grab datasets.",
    )
    parser.add_argument(
        "--default-fps",
        type=float,
        default=30.0,
        help="Fallback FPS when duration cannot be inferred from files.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-demo statistics in addition to task totals.",
    )
    return parser.parse_args()


def is_demo_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name.endswith(SKIP_SUFFIXES):
        return False
    return any(path.glob("*.pkl"))


def find_demo_dirs(task_dir: Path) -> list[Path]:
    demo_dirs = [item for item in sorted(task_dir.iterdir()) if is_demo_dir(item)]
    if demo_dirs:
        return demo_dirs
    if any(task_dir.glob("*.pkl")):
        return [task_dir]
    return []


def parse_timestamp_from_pkl(pkl_path: Path) -> dt.datetime | None:
    stem = pkl_path.stem
    if not TIMESTAMP_RE.match(stem):
        return None
    # Stored filenames use ISO timestamps with ":" replaced by "-".
    parts = stem.split("T")
    if len(parts) != 2:
        return None
    date_part, time_part = parts
    time_chunks = time_part.split("-")
    if len(time_chunks) < 3:
        return None
    hh, mm, ss = time_chunks[:3]
    micros = time_chunks[3] if len(time_chunks) > 3 else "0"
    try:
        return dt.datetime.fromisoformat(
            f"{date_part}T{hh}:{mm}:{ss}.{micros}"
        )
    except ValueError:
        return None


def read_average_fps(freq_path: Path) -> float | None:
    if not freq_path.exists():
        return None
    try:
        content = freq_path.read_text()
    except OSError:
        return None
    match = FPS_RE.search(content)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def estimate_demo_duration_seconds(demo_dir: Path, pkl_files: list[Path], default_fps: float) -> tuple[float, str]:
    if not pkl_files:
        return 0.0, "empty"

    first_ts = parse_timestamp_from_pkl(pkl_files[0])
    last_ts = parse_timestamp_from_pkl(pkl_files[-1])
    if first_ts is not None and last_ts is not None and last_ts >= first_ts:
        return (last_ts - first_ts).total_seconds(), "timestamps"

    avg_fps = read_average_fps(demo_dir / "freq.txt")
    if avg_fps and avg_fps > 0:
        return len(pkl_files) / avg_fps, "freq.txt"

    return len(pkl_files) / default_fps, f"default_fps={default_fps:g}"


def analyze_demo(demo_dir: Path, default_fps: float) -> DemoStats:
    pkl_files = sorted(demo_dir.glob("*.pkl"))
    seconds, source = estimate_demo_duration_seconds(demo_dir, pkl_files, default_fps)
    return DemoStats(
        demo_dir=demo_dir,
        frames=len(pkl_files),
        seconds=seconds,
        duration_source=source,
    )


def analyze_task(task_dir: Path, default_fps: float) -> TaskStats:
    demo_dirs = find_demo_dirs(task_dir)
    demos = [analyze_demo(demo_dir, default_fps) for demo_dir in demo_dirs]
    return TaskStats(task_dir=task_dir, demos=demos)


def print_task_summary(task_stats: TaskStats) -> None:
    total_minutes = task_stats.total_seconds / 60.0
    avg_minutes = task_stats.avg_seconds / 60.0
    print(f"Task: {task_stats.task_dir}")
    print(f"  num demos per task: {task_stats.num_demos}")
    print(f"  total frames per task: {task_stats.total_frames}")
    print(f"  total minutes per task: {total_minutes:.2f}")
    print(f"  average frames per demo: {task_stats.avg_frames:.2f}")
    print(f"  average minutes per demo: {avg_minutes:.2f}")


def print_demo_details(task_stats: TaskStats) -> None:
    if not task_stats.demos:
        print("  no demos found")
        return
    print("  demo details:")
    for demo in task_stats.demos:
        print(
            f"    {demo.demo_dir.name}: "
            f"{demo.frames} frames, "
            f"{demo.seconds / 60.0:.2f} min, "
            f"duration_source={demo.duration_source}"
        )


def main() -> None:
    args = parse_args()

    all_stats: list[TaskStats] = []
    missing_paths: list[Path] = []

    for dataset_path in args.dataset_paths:
        task_dir = Path(dataset_path).expanduser()
        if not task_dir.exists():
            missing_paths.append(task_dir)
            continue
        all_stats.append(analyze_task(task_dir, args.default_fps))

    if missing_paths:
        print("Missing dataset paths:")
        for path in missing_paths:
            print(f"  {path}")
        print()

    if not all_stats:
        print("No valid dataset paths found.")
        return

    print("=" * 72)
    print("Task Dataset Statistics")
    print("=" * 72)
    for task_stats in all_stats:
        print_task_summary(task_stats)
        if args.details:
            print_demo_details(task_stats)
        print()

    print("=" * 72)
    print("Summary Table")
    print("=" * 72)
    print(
        f"{'task':30s} {'demos':>8s} {'total_frames':>14s} "
        f"{'total_min':>12s} {'avg_frames':>12s} {'avg_min':>10s}"
    )
    for task_stats in all_stats:
        print(
            f"{task_stats.task_dir.name:30s} "
            f"{task_stats.num_demos:8d} "
            f"{task_stats.total_frames:14d} "
            f"{task_stats.total_seconds / 60.0:12.2f} "
            f"{task_stats.avg_frames:12.2f} "
            f"{task_stats.avg_seconds / 60.0:10.2f}"
        )


if __name__ == "__main__":
    main()
