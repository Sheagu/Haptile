"""
Data Analysis Tool - 分析保存的数据集统计信息
"""

import argparse
import glob
import pickle
from pathlib import Path

# Fix numpy compatibility issue
import numpy as np
if not hasattr(np, '_core'):
    np._core = np.core

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt


def analyze_dataset(data_dir: str):
    """分析数据集"""
    data_dir = Path(data_dir)
    pkl_files = sorted(glob.glob(str(data_dir / "*.pkl")))
    
    if len(pkl_files) == 0:
        print(f"No pkl files found in {data_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"Dataset Analysis: {data_dir}")
    print(f"{'='*60}\n")
    
    print(f"Total frames: {len(pkl_files)}")
    
    # 加载所有数据
    all_joint_positions = []
    all_ee_poses = []
    all_controls = []
    
    print("Loading data...")
    for i, pkl_file in enumerate(pkl_files):
        if i % 50 == 0:
            print(f"  {i}/{len(pkl_files)}")
        
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
        
        if "joint_positions" in data:
            all_joint_positions.append(data["joint_positions"])
        
        if "ee_pos_quat" in data:
            all_ee_poses.append(data["ee_pos_quat"])
        
        if "control" in data:
            all_controls.append(data["control"])
    
    # 统计信息
    print(f"\n{'='*60}")
    print("Statistics")
    print(f"{'='*60}\n")
    
    if len(all_joint_positions) > 0:
        all_joint_positions = np.array(all_joint_positions)
        print(f"Joint Positions Shape: {all_joint_positions.shape}")
        print(f"  Mean: {np.mean(all_joint_positions, axis=0)}")
        print(f"  Std:  {np.std(all_joint_positions, axis=0)}")
        print(f"  Min:  {np.min(all_joint_positions, axis=0)}")
        print(f"  Max:  {np.max(all_joint_positions, axis=0)}")
        print()
    
    if len(all_ee_poses) > 0:
        all_ee_poses = np.array(all_ee_poses)
        print(f"End-Effector Poses Shape: {all_ee_poses.shape}")
        print(f"  Position:")
        print(f"    Mean: {np.mean(all_ee_poses[:, :3], axis=0)}")
        print(f"    Std:  {np.std(all_ee_poses[:, :3], axis=0)}")
        print(f"    Min:  {np.min(all_ee_poses[:, :3], axis=0)}")
        print(f"    Max:  {np.max(all_ee_poses[:, :3], axis=0)}")
        print()
    
    if len(all_controls) > 0:
        all_controls = np.array(all_controls)
        print(f"Control Commands Shape: {all_controls.shape}")
        print(f"  Mean: {np.mean(all_controls, axis=0)}")
        print(f"  Std:  {np.std(all_controls, axis=0)}")
        print(f"  Min:  {np.min(all_controls, axis=0)}")
        print(f"  Max:  {np.max(all_controls, axis=0)}")
        print()
    
    # 绘制图表
    if len(all_joint_positions) > 0:
        plot_trajectories(all_joint_positions, all_ee_poses, all_controls, data_dir)


def plot_trajectories(joint_positions, ee_poses, controls, data_dir):
    """绘制轨迹图"""
    num_joints = joint_positions.shape[1]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f"Trajectory Analysis - {data_dir.name}", fontsize=16)
    
    # 关节位置
    ax = axes[0]
    for i in range(min(num_joints, 7)):  # 最多显示7个关节
        ax.plot(joint_positions[:, i], label=f"Joint {i}", alpha=0.7)
    ax.set_ylabel("Joint Position (rad)")
    ax.set_title("Joint Positions Over Time")
    ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5))
    ax.grid(True, alpha=0.3)
    
    # 末端执行器位置
    if len(ee_poses) > 0:
        ax = axes[1]
        ee_poses = np.array(ee_poses)
        ax.plot(ee_poses[:, 0], label="X", alpha=0.7)
        ax.plot(ee_poses[:, 1], label="Y", alpha=0.7)
        ax.plot(ee_poses[:, 2], label="Z", alpha=0.7)
        ax.set_ylabel("Position (m)")
        ax.set_title("End-Effector Position Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 控制指令
    if len(controls) > 0:
        ax = axes[2]
        controls = np.array(controls)
        for i in range(min(controls.shape[1], 7)):
            ax.plot(controls[:, i], label=f"Control {i}", alpha=0.7)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Control Command")
        ax.set_title("Control Commands Over Time")
        ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = data_dir / "trajectory_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nTrajectory plot saved to: {output_path}")
    
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze TeleUR dataset")
    parser.add_argument("data_dir", type=str, help="Path to data directory")
    
    args = parser.parse_args()
    analyze_dataset(args.data_dir)


if __name__ == "__main__":
    main()
