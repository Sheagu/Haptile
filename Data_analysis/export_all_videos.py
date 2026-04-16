#!/usr/bin/env python3
"""
批量导出数据集视频工具
Export videos for all datasets in a folder

This tool allows you to batch export videos from all datasets in a directory.
It supports:
- Exporting all datasets or only the N most recent ones
- Skipping already exported videos to save time
- Custom output directory and frame rate
- Progress tracking and error handling
- Listing all available datasets

Usage:
    # Export all datasets
    python export_all_videos.py ./shared/data/bc_data
    
    # Export only the 5 most recent datasets
    python export_all_videos.py ./shared/data/bc_data --recent 5
    
    # List all datasets without exporting
    python export_all_videos.py ./shared/data/bc_data --list
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Import the visualizer
from visualize_data import DataVisualizer


def export_all_videos(
    data_root: str,
    output_dir: str = "videos",
    fps: int = 10,
    skip_existing: bool = True,
    verbose: bool = True,
):
    """
    批量导出所有数据集的视频
    Export videos for all datasets in a folder
    
    Args:
        data_root: 数据根目录 (例如 ./shared/data/bc_data)
        output_dir: 输出视频目录
        fps: 视频帧率
        skip_existing: 是否跳过已存在的视频
        verbose: 是否显示详细输出
    
    Returns:
        Dictionary with statistics: {total, success, failed, skipped, duration}
    """
    start_time = time.time()
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有数据集文件夹
    dataset_dirs = []
    for item in data_root.iterdir():
        if item.is_dir():
            # 检查是否包含pkl文件
            pkl_files = list(item.glob("*.pkl"))
            if len(pkl_files) > 0:
                dataset_dirs.append((item, len(pkl_files)))
    
    if len(dataset_dirs) == 0:
        print(f"❌ No dataset folders found in {data_root}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0, "duration": 0}
    
    # 按时间排序 (最旧的在前)
    dataset_dirs.sort(key=lambda x: x[0].stat().st_mtime)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Batch Video Export")
        print(f"{'='*60}")
        print(f"Data root: {data_root}")
        print(f"Output dir: {output_dir}")
        print(f"Found {len(dataset_dirs)} datasets")
        print(f"FPS: {fps}")
        print(f"Skip existing: {skip_existing}")
        print(f"{'='*60}\n")
    
    # 统计
    total = len(dataset_dirs)
    success = 0
    failed = 0
    skipped = 0
    errors = []
    
    # 逐个导出
    for i, (dataset_dir, num_frames) in enumerate(dataset_dirs, 1):
        dataset_name = dataset_dir.name
        output_path = output_dir / f"{dataset_name}.mp4"
        
        if verbose:
            print(f"\n[{i}/{total}] Processing: {dataset_name} ({num_frames} frames)")
            print(f"  Output: {output_path}")
        
        # 检查是否已存在
        if skip_existing and output_path.exists():
            if verbose:
                file_size = output_path.stat().st_size / 1024 / 1024  # MB
                print(f"  ⏭️  Skipped (already exists, {file_size:.1f} MB)")
            skipped += 1
            continue
        
        try:
            # 创建可视化器
            visualizer = DataVisualizer(str(dataset_dir))
            
            # 导出视频
            export_start = time.time()
            visualizer.export_video(str(output_path), fps=fps)
            export_duration = time.time() - export_start
            
            if verbose:
                file_size = output_path.stat().st_size / 1024 / 1024  # MB
                print(f"  ✅ Success! ({file_size:.1f} MB, {export_duration:.1f}s)")
            success += 1
            
        except Exception as e:
            error_msg = f"{dataset_name}: {str(e)}"
            errors.append(error_msg)
            if verbose:
                print(f"  ❌ Failed: {e}")
            failed += 1
            continue
    
    # 计算总时间
    total_duration = time.time() - start_time
    
    # 打印总结
    if verbose:
        print(f"\n{'='*60}")
        print(f"Summary")
        print(f"{'='*60}")
        print(f"Total datasets: {total}")
        print(f"✅ Success: {success}")
        print(f"⏭️  Skipped: {skipped}")
        print(f"❌ Failed: {failed}")
        print(f"⏱️  Duration: {total_duration:.1f}s")
        print(f"\nVideos saved to: {output_dir.absolute()}")
        
        if errors:
            print(f"\n⚠️  Errors:")
            for error in errors:
                print(f"  - {error}")
        
        print(f"{'='*60}\n")
    
    return {
        "total": total,
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "duration": total_duration,
        "errors": errors
    }


def export_recent_videos(
    data_root: str,
    output_dir: str = "videos",
    num_recent: int = 5,
    fps: int = 10,
    skip_existing: bool = True,
):
    """
    导出最近N个数据集的视频
    Export videos for the N most recent datasets
    
    Args:
        data_root: 数据根目录
        output_dir: 输出视频目录
        num_recent: 导出最近几个数据集
        fps: 视频帧率
        skip_existing: 是否跳过已存在的视频
    
    Returns:
        Dictionary with statistics
    """
    start_time = time.time()
    data_root = Path(data_root)
    output_dir = Path(output_dir)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有数据集文件夹
    dataset_dirs = []
    for item in data_root.iterdir():
        if item.is_dir():
            pkl_files = list(item.glob("*.pkl"))
            if len(pkl_files) > 0:
                dataset_dirs.append((item, len(pkl_files)))
    
    if len(dataset_dirs) == 0:
        print(f"❌ No dataset folders found in {data_root}")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0, "duration": 0}
    
    # 按修改时间排序，取最新的N个
    dataset_dirs.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    original_count = len(dataset_dirs)
    dataset_dirs = dataset_dirs[:num_recent]
    
    print(f"\n{'='*60}")
    print(f"Export Recent {num_recent} Datasets")
    print(f"{'='*60}")
    print(f"Total available: {original_count}")
    print(f"Exporting: {len(dataset_dirs)}")
    print(f"Data root: {data_root}")
    print(f"Output dir: {output_dir}")
    print(f"FPS: {fps}")
    print(f"Skip existing: {skip_existing}")
    print(f"{'='*60}\n")
    
    # 统计
    success = 0
    failed = 0
    skipped = 0
    errors = []
    
    # 逐个导出
    for i, (dataset_dir, num_frames) in enumerate(dataset_dirs, 1):
        dataset_name = dataset_dir.name
        output_path = output_dir / f"{dataset_name}.mp4"
        
        mtime = datetime.fromtimestamp(dataset_dir.stat().st_mtime)
        print(f"\n[{i}/{len(dataset_dirs)}] {dataset_name} ({num_frames} frames, {mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        print(f"  Output: {output_path}")
        
        # 检查是否已存在
        if skip_existing and output_path.exists():
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            print(f"  ⏭️  Skipped (already exists, {file_size:.1f} MB)")
            skipped += 1
            continue
        
        try:
            export_start = time.time()
            visualizer = DataVisualizer(str(dataset_dir))
            visualizer.export_video(str(output_path), fps=fps)
            export_duration = time.time() - export_start
            
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            print(f"  ✅ Success! ({file_size:.1f} MB, {export_duration:.1f}s)")
            success += 1
        except Exception as e:
            error_msg = f"{dataset_name}: {str(e)}"
            errors.append(error_msg)
            print(f"  ❌ Failed: {e}")
            failed += 1
    
    # 计算总时间
    total_duration = time.time() - start_time
    
    # 打印总结
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"✅ Success: {success}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Failed: {failed}")
    print(f"⏱️  Duration: {total_duration:.1f}s")
    print(f"\nVideos saved to: {output_dir.absolute()}")
    
    if errors:
        print(f"\n⚠️  Errors:")
        for error in errors:
            print(f"  - {error}")
    
    print(f"{'='*60}\n")
    
    return {
        "total": len(dataset_dirs),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "duration": total_duration,
        "errors": errors
    }


def list_datasets(data_root: str):
    """列出所有数据集"""
    data_root = Path(data_root)
    
    # 查找所有数据集文件夹
    dataset_dirs = []
    for item in data_root.iterdir():
        if item.is_dir():
            pkl_files = list(item.glob("*.pkl"))
            if len(pkl_files) > 0:
                dataset_dirs.append((item, len(pkl_files)))
    
    if len(dataset_dirs) == 0:
        print(f"❌ No dataset folders found in {data_root}")
        return
    
    # 按时间排序
    dataset_dirs.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    
    print(f"\n{'='*60}")
    print(f"Datasets in {data_root}")
    print(f"{'='*60}\n")
    
    for i, (dataset_dir, num_frames) in enumerate(dataset_dirs, 1):
        mtime = datetime.fromtimestamp(dataset_dir.stat().st_mtime)
        print(f"{i:3d}. {dataset_dir.name:20s} | {num_frames:4d} frames | {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{'='*60}")
    print(f"Total: {len(dataset_dirs)} datasets\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch export videos from TeleUR datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 导出所有数据集的视频
  python export_all_videos.py ./shared/data/bc_data
  
  # 导出到指定目录，设置帧率
  python export_all_videos.py ./shared/data/bc_data --output-dir my_videos --fps 15
  
  # 只导出最近5个数据集
  python export_all_videos.py ./shared/data/bc_data --recent 5
  
  # 列出所有数据集
  python export_all_videos.py ./shared/data/bc_data --list
  
  # 强制重新导出（覆盖已存在的视频）
  python export_all_videos.py ./shared/data/bc_data --no-skip-existing
        """
    )
    
    parser.add_argument(
        "data_root",
        type=str,
        help="Root directory containing dataset folders (e.g., ./shared/data/bc_data)"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="videos",
        help="Output directory for videos (default: videos)"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Video frame rate (default: 10)"
    )
    
    parser.add_argument(
        "--recent",
        type=int,
        default=None,
        help="Only export N most recent datasets"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all datasets and exit"
    )
    
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-export even if video already exists"
    )
    
    args = parser.parse_args()
    
    # 列出数据集
    if args.list:
        list_datasets(args.data_root)
        return
    
    # 导出最近N个数据集
    if args.recent is not None:
        export_recent_videos(
            args.data_root,
            args.output_dir,
            args.recent,
            args.fps,
            skip_existing=not args.no_skip_existing
        )
    else:
        # 导出所有数据集
        export_all_videos(
            args.data_root,
            args.output_dir,
            args.fps,
            skip_existing=not args.no_skip_existing,
            verbose=True
        )


if __name__ == "__main__":
    main()
