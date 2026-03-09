"""
Data Visualization Tool for TeleUR Data
可视化保存的机器人遥操作数据，包括RGB图像、深度图、触觉传感器等
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

# Fix numpy compatibility issue
import numpy as np
# Handle numpy version compatibility for pickle
if not hasattr(np, '_core'):
    np._core = np.core

import cv2
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better compatibility
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


class DataVisualizer:
    def __init__(self, data_dir: str):
        """初始化数据可视化器
        
        Args:
            data_dir: 数据目录路径
        """
        self.data_dir = Path(data_dir)
        
        # 查找所有pkl文件
        self.pkl_files = sorted(glob.glob(str(self.data_dir / "*.pkl")))
        
        if len(self.pkl_files) == 0:
            raise ValueError(f"No pkl files found in {data_dir}")
        
        print(f"Found {len(self.pkl_files)} frames in {data_dir}")
        
        # 加载第一个有效的帧以获取信息
        first_frame = None
        for i, pkl_file in enumerate(self.pkl_files):
            try:
                with open(pkl_file, "rb") as f:
                    first_frame = pickle.load(f)
                break  # Found first valid frame
            except (EOFError, pickle.UnpicklingError) as e:
                if i == 0:
                    print(f"Warning: First frame {pkl_file} is corrupted, trying next frames...")
                continue
        
        if first_frame is None:
            print("Error: No valid frames found in the dataset!")
            sys.exit(1)
        
        # 检测可用的数据类型
        self.has_base_rgb = "base_camera_rgb" in first_frame
        self.has_base_depth = "base_camera_depth" in first_frame
        self.has_tactile_left = "tactile_left_rgb" in first_frame
        self.has_tactile_right = "tactile_right_rgb" in first_frame
        self.has_joint_positions = "joint_positions" in first_frame
        self.has_ee_pose = "ee_pos_quat" in first_frame
        self.has_control = "control" in first_frame
        
        print("\n=== Data Available ===")
        print(f"Base Camera RGB: {self.has_base_rgb}")
        print(f"Base Camera Depth: {self.has_base_depth}")
        print(f"Tactile Left: {self.has_tactile_left}")
        print(f"Tactile Right: {self.has_tactile_right}")
        print(f"Joint Positions: {self.has_joint_positions}")
        print(f"End-Effector Pose: {self.has_ee_pose}")
        print(f"Control Commands: {self.has_control}")
        
        if self.has_base_rgb:
            rgb_shape = first_frame["base_camera_rgb"].shape
            print(f"\nBase RGB Shape: {rgb_shape}")
            if len(rgb_shape) == 4:
                self.num_base_cameras = rgb_shape[0]
                print(f"Number of base cameras: {self.num_base_cameras}")
            else:
                self.num_base_cameras = 1
        
        self.current_frame = 0
        self.playing = False
        
    def load_frame(self, frame_idx: int):
        """加载指定帧的数据"""
        with open(self.pkl_files[frame_idx], "rb") as f:
            try:
                return pickle.load(f)
            except EOFError as e:
                print(f"Warning: Corrupted pickle file {self.pkl_files[frame_idx]}")
                raise
    
    def normalize_depth(self, depth):
        """归一化深度图用于显示"""
        if depth is None or depth.size == 0:
            return None
        depth_normalized = depth.astype(np.float32)
        # 移除0值（无效深度）
        valid_depth = depth_normalized[depth_normalized > 0]
        if len(valid_depth) == 0:
            return np.zeros_like(depth, dtype=np.uint8)
        
        # 使用百分位数进行归一化，避免异常值影响
        min_val = np.percentile(valid_depth, 1)
        max_val = np.percentile(valid_depth, 99)
        
        depth_normalized = np.clip(depth_normalized, min_val, max_val)
        depth_normalized = (depth_normalized - min_val) / (max_val - min_val + 1e-6)
        return (depth_normalized * 255).astype(np.uint8)
    
    def visualize_interactive(self):
        """交互式可视化"""
        # 计算子图数量
        num_plots = 0
        if self.has_base_rgb:
            num_plots += self.num_base_cameras
        if self.has_base_depth:
            num_plots += self.num_base_cameras
        if self.has_tactile_left:
            num_plots += 1
        if self.has_tactile_right:
            num_plots += 1
        
        if num_plots == 0:
            print("No image data to visualize!")
            return
        
        # 创建图形
        cols = min(3, num_plots)
        rows = (num_plots + cols - 1) // cols
        
        fig = plt.figure(figsize=(6*cols, 5*rows + 2))
        fig.suptitle(f"TeleUR Data Visualization - {self.data_dir.name}", fontsize=16)
        
        # 创建子图
        axes = []
        plot_idx = 0
        
        # Base camera RGB
        if self.has_base_rgb:
            for i in range(self.num_base_cameras):
                ax = plt.subplot(rows, cols, plot_idx + 1)
                axes.append(ax)
                ax.set_title(f"Base Camera {i} - RGB")
                ax.axis('off')
                plot_idx += 1
        
        # Base camera Depth
        # if self.has_base_depth:
        #     for i in range(self.num_base_cameras):
        #         ax = plt.subplot(rows, cols, plot_idx + 1)
        #         axes.append(ax)
        #         ax.set_title(f"Base Camera {i} - Depth")
        #         ax.axis('off')
        #         plot_idx += 1
        
        # Tactile sensors
        if self.has_tactile_left:
            ax = plt.subplot(rows, cols, plot_idx + 1)
            axes.append(ax)
            ax.set_title("Tactile Left")
            ax.axis('off')
            plot_idx += 1
        
        if self.has_tactile_right:
            ax = plt.subplot(rows, cols, plot_idx + 1)
            axes.append(ax)
            ax.set_title("Tactile Right")
            ax.axis('off')
            plot_idx += 1
        
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        
        # 创建滑块
        ax_slider = plt.axes([0.15, 0.08, 0.65, 0.03])
        slider = Slider(ax_slider, 'Frame', 0, len(self.pkl_files)-1, 
                       valinit=0, valstep=1)
        
        # 创建播放按钮
        ax_play = plt.axes([0.15, 0.02, 0.1, 0.04])
        btn_play = Button(ax_play, 'Play/Pause')
        
        # 创建上一帧/下一帧按钮
        ax_prev = plt.axes([0.3, 0.02, 0.08, 0.04])
        btn_prev = Button(ax_prev, 'Prev')
        
        ax_next = plt.axes([0.4, 0.02, 0.08, 0.04])
        btn_next = Button(ax_next, 'Next')
        
        # 文本显示区域
        info_text = fig.text(0.82, 0.08, '', fontsize=10, verticalalignment='top',
                            family='monospace', bbox=dict(boxstyle='round', 
                            facecolor='wheat', alpha=0.5))
        
        def update(val):
            """更新显示"""
            frame_idx = int(slider.val)
            data = self.load_frame(frame_idx)
            
            plot_idx = 0
            
            # 更新Base RGB
            if self.has_base_rgb:
                rgb = data["base_camera_rgb"]
                if len(rgb.shape) == 4:  # Multiple cameras
                    for i in range(self.num_base_cameras):
                        axes[plot_idx].clear()
                        axes[plot_idx].imshow(rgb[i])
                        axes[plot_idx].set_title(f"Base Camera {i} - RGB")
                        axes[plot_idx].axis('off')
                        plot_idx += 1
                else:  # Single camera
                    axes[plot_idx].clear()
                    axes[plot_idx].imshow(rgb)
                    axes[plot_idx].set_title("Base Camera - RGB")
                    axes[plot_idx].axis('off')
                    plot_idx += 1
            
            # 更新Base Depth
            if self.has_base_depth:
                depth = data["base_camera_depth"]
                if len(depth.shape) == 3:  # Multiple cameras
                    for i in range(self.num_base_cameras):
                        axes[plot_idx].clear()
                        depth_vis = self.normalize_depth(depth[i])
                        axes[plot_idx].imshow(depth_vis, cmap='jet')
                        axes[plot_idx].set_title(f"Base Camera {i} - Depth")
                        axes[plot_idx].axis('off')
                        plot_idx += 1
                else:  # Single camera
                    axes[plot_idx].clear()
                    depth_vis = self.normalize_depth(depth)
                    axes[plot_idx].imshow(depth_vis, cmap='jet')
                    axes[plot_idx].set_title("Base Camera - Depth")
                    axes[plot_idx].axis('off')
                    plot_idx += 1
            
            # 更新Tactile Left
            if self.has_tactile_left:
                tactile = data["tactile_left_rgb"]
                if tactile.ndim == 4 and tactile.shape[0] == 1:
                    tactile = tactile[0]
                axes[plot_idx].clear()
                axes[plot_idx].imshow(tactile)
                axes[plot_idx].set_title("Tactile Left")
                axes[plot_idx].axis('off')
                plot_idx += 1
            
            # 更新Tactile Right
            if self.has_tactile_right:
                tactile = data["tactile_right_rgb"]
                if tactile.ndim == 4 and tactile.shape[0] == 1:
                    tactile = tactile[0]
                axes[plot_idx].clear()
                axes[plot_idx].imshow(tactile)
                axes[plot_idx].set_title("Tactile Right")
                axes[plot_idx].axis('off')
                plot_idx += 1
            
            # 更新信息文本
            info_str = f"Frame: {frame_idx}/{len(self.pkl_files)-1}\n"
            info_str += f"File: {os.path.basename(self.pkl_files[frame_idx])}\n\n"
            
            if self.has_joint_positions:
                joints = data["joint_positions"]
                # Convert to numpy array if it's a list
                if not isinstance(joints, np.ndarray):
                    joints = np.array(joints)
                info_str += f"Joint Positions:\n"
                info_str += f"  {np.array2string(joints, precision=3, suppress_small=True)}\n\n"
            
            if self.has_control:
                control = data["control"]
                # Convert to numpy array if it's a list
                if not isinstance(control, np.ndarray):
                    control = np.array(control)
                info_str += f"Control Command:\n"
                info_str += f"  {np.array2string(control, precision=3, suppress_small=True)}\n\n"
            
            if self.has_ee_pose:
                ee_pose = data["ee_pos_quat"]
                # Convert to numpy array if it's a list
                if not isinstance(ee_pose, np.ndarray):
                    ee_pose = np.array(ee_pose)
                info_str += f"End-Effector Pose:\n"
                info_str += f"  Pos: {np.array2string(ee_pose[:3], precision=3)}\n"
                if len(ee_pose) > 3:
                    info_str += f"  Quat: {np.array2string(ee_pose[3:], precision=3)}\n"
            
            info_text.set_text(info_str)
            
            fig.canvas.draw_idle()
        
        def play_pause(event):
            """播放/暂停"""
            self.playing = not self.playing
            if self.playing:
                play_animation()
        
        def play_animation():
            """播放动画"""
            if self.playing:
                current = int(slider.val)
                if current < len(self.pkl_files) - 1:
                    slider.set_val(current + 1)
                    fig.canvas.draw_idle()
                    fig.canvas.start_event_loop(0.033)  # ~30 FPS
                    play_animation()
                else:
                    self.playing = False
        
        def prev_frame(event):
            """上一帧"""
            current = int(slider.val)
            if current > 0:
                slider.set_val(current - 1)
        
        def next_frame(event):
            """下一帧"""
            current = int(slider.val)
            if current < len(self.pkl_files) - 1:
                slider.set_val(current + 1)
        
        # 绑定事件
        slider.on_changed(update)
        btn_play.on_clicked(play_pause)
        btn_prev.on_clicked(prev_frame)
        btn_next.on_clicked(next_frame)
        
        # 初始化显示
        update(0)
        
        plt.show()
    
    def export_video(self, output_path: str, fps: int = 10):
        """导出视频
        
        Args:
            output_path: 输出视频路径
            fps: 帧率
        """
        print(f"Exporting video to {output_path}...")
        
        # 加载第一个有效的帧获取尺寸信息
        first_data = None
        for i in range(len(self.pkl_files)):
            try:
                first_data = self.load_frame(i)
                if first_data is not None:
                    break
            except (EOFError, pickle.UnpicklingError):
                continue
        
        if first_data is None:
            print("Error: No valid frames found to export!")
            return
        
        # 收集所有图像
        images = []
        corrupted_frames = []
        for i, pkl_file in enumerate(self.pkl_files):
            if i % 10 == 0:
                print(f"Processing frame {i}/{len(self.pkl_files)}")
            
            try:
                data = self.load_frame(i)
            except (EOFError, pickle.UnpicklingError) as e:
                print(f"Skipping corrupted frame {i}: {pkl_file}")
                corrupted_frames.append((i, pkl_file))
                continue
            
            # 创建可视化图像
            img_list = []
            
            # Base RGB
            if self.has_base_rgb:
                rgb = data["base_camera_rgb"]
                if len(rgb.shape) == 4:
                    for j in range(self.num_base_cameras):
                        img_list.append(rgb[j])
                else:
                    img_list.append(rgb)
            
            # Tactile sensors
            if self.has_tactile_left:
                tactile = data["tactile_left_rgb"]
                if tactile.ndim == 4 and tactile.shape[0] == 1:
                    tactile = tactile[0]
                img_list.append(tactile)
            
            if self.has_tactile_right:
                tactile = data["tactile_right_rgb"]
                if tactile.ndim == 4 and tactile.shape[0] == 1:
                    tactile = tactile[0]
                img_list.append(tactile)
            
            # 水平拼接图像
            if len(img_list) > 0:
                # 确保所有图像高度一致
                max_height = max(img.shape[0] for img in img_list)
                resized_imgs = []
                for img in img_list:
                    if img.shape[0] != max_height:
                        scale = max_height / img.shape[0]
                        new_width = int(img.shape[1] * scale)
                        img = cv2.resize(img, (new_width, max_height))
                    resized_imgs.append(img)
                
                combined = np.concatenate(resized_imgs, axis=1)
                
                # 添加帧号
                combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
                cv2.putText(combined, f"Frame: {i}/{len(self.pkl_files)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
                           (255, 255, 255), 2)
                
                images.append(combined)
        
        if corrupted_frames:
            print(f"\nWarning: {len(corrupted_frames)} corrupted frames were skipped:")
            for frame_idx, file_path in corrupted_frames[:10]:  # Show first 10
                print(f"  Frame {frame_idx}: {file_path}")
            if len(corrupted_frames) > 10:
                print(f"  ... and {len(corrupted_frames) - 10} more")
        # 写入视频
        if len(images) > 0:
            height, width = images[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            for img in images:
                out.write(img)
            
            out.release()
            print(f"Video exported successfully to {output_path}")
        else:
            print("No images to export!")


def main():
    parser = argparse.ArgumentParser(description="Visualize TeleUR collected data")
    parser.add_argument("data_dir", type=str, help="Path to data directory containing pkl files")
    parser.add_argument("--export-video", type=str, default=None, 
                       help="Export video to specified path (e.g., output.mp4)")
    parser.add_argument("--fps", type=int, default=10, help="FPS for exported video")
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = DataVisualizer(args.data_dir)
    
    if args.export_video:
        # 导出视频
        visualizer.export_video(args.export_video, fps=args.fps)
    else:
        # 交互式可视化
        visualizer.visualize_interactive()


if __name__ == "__main__":
    main()
