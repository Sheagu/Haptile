# 数据可视化工具 - 快速开始

## 📦 已创建的工具

| 工具 | 功能 | 使用场景 |
|-----|------|---------|
| `visualize_data.py` | 交互式数据查看器 | 查看图像、传感器数据、播放动画 |
| `analyze_data.py` | 统计分析工具 | 分析轨迹、生成统计图表 |
| `test_visualization.sh` | 测试脚本 | 快速验证工具是否正常工作 |

## 🚀 快速开始

### 1. 查看数据（交互式）
```bash
python visualize_data.py ./shared/data/bc_data/YOUR_DATA_FOLDER
```

**界面说明：**
- 📸 **左侧/上方**: 所有摄像头图像（base camera + tactile sensors）
- 📊 **右侧**: 传感器数据（关节位置、控制指令等）
- 🎚️ **底部滑块**: 选择帧
- 🎮 **按钮**: Play/Pause、Prev、Next

### 2. 分析数据
```bash
python analyze_data.py ./shared/data/bc_data/YOUR_DATA_FOLDER
```

**输出：**
- 终端显示统计信息
- 生成 `trajectory_analysis.png` 图表

### 3. 导出视频
```bash
python visualize_data.py ./shared/data/bc_data/YOUR_DATA_FOLDER \
  --export-video output.mp4 \
  --fps 10
```

## 📖 详细文档

- **[DATA_VISUALIZATION_GUIDE.md](DATA_VISUALIZATION_GUIDE.md)** - 完整使用指南
- **[QUICK_START.md](QUICK_START.md)** - 数据采集快速指南  
- **[TACTILE_CAMERA_SETUP.md](TACTILE_CAMERA_SETUP.md)** - 相机配置指南

## 💡 常用命令速查

```bash
# 查看最新采集的数据
python visualize_data.py ./shared/data/bc_data/$(ls -t ./shared/data/bc_data/ | head -1)

# 分析所有数据集
for dir in ./shared/data/bc_data/*/; do
    python analyze_data.py "$dir"
done

# 为每个数据集创建视频
for dir in ./shared/data/bc_data/*/; do
    dirname=$(basename "$dir")
    python visualize_data.py "$dir" --export-video "videos/${dirname}.mp4" --fps 10
done

# 测试工具
./test_visualization.sh ./shared/data/bc_data/YOUR_DATA_FOLDER
```

## 📋 数据文件说明

每个采集会话保存在独立的文件夹中，包含：

```
0127_154230/                           # 采集时间戳
├── 2026-01-27T15-42-30-123456.pkl    # 每一帧的数据
├── 2026-01-27T15-42-30-156789.pkl
├── ...
├── freq.txt                           # 采集频率统计
└── trajectory_analysis.png            # 运行analyze_data.py后生成
```

### PKL文件内容
```python
{
    'base_camera_rgb': np.ndarray,     # (N, H, W, 3) 或 (H, W, 3)
    'base_camera_depth': np.ndarray,   # (N, H, W) 或 (H, W)
    'tactile_left_rgb': np.ndarray,    # (H, W, 3) - 如果启用
    'tactile_right_rgb': np.ndarray,   # (H, W, 3) - 如果启用
    'joint_positions': np.ndarray,     # (7,) 关节角度
    'ee_pos_quat': np.ndarray,        # (6,) 末端位姿
    'control': np.ndarray,             # 控制指令
    'activated': bool                  # 是否激活
}
```

## 🔧 依赖安装

如果遇到导入错误，安装以下包：

```bash
pip install numpy matplotlib opencv-python
sudo apt-get install python3-tk  # for matplotlib GUI
```

## 📸 示例截图

### 交互式查看器
![Interactive Viewer](docs/interactive_viewer.png)
- 多摄像头同时显示
- 实时传感器数据
- 播放控制

### 轨迹分析
![Trajectory Analysis](docs/trajectory_analysis.png)
- 关节位置变化
- 末端执行器轨迹
- 控制指令可视化

## ⚡ 性能提示

1. **大数据集**: 如果数据集很大（>1000帧），加载可能较慢
   - 解决：在代码中添加采样 `pkl_files[::5]` (每5帧取1帧)

2. **视频导出**: 导出高分辨率视频需要时间和空间
   - 建议：先用低fps测试 `--fps 5`

3. **内存占用**: 同时显示多个高分辨率图像会占用内存
   - 解决：关闭不需要的图像显示

## 🐛 故障排除

### 问题: "No module named 'tkinter'"
```bash
sudo apt-get install python3-tk
```

### 问题: matplotlib无法显示窗口
```bash
# 在代码开头添加
import matplotlib
matplotlib.use('TkAgg')
```

### 问题: 视频导出失败
```bash
sudo apt-get install ffmpeg
```

## 📞 需要帮助？

1. 查看详细文档: `DATA_VISUALIZATION_GUIDE.md`
2. 运行测试脚本: `./test_visualization.sh <data_dir>`
3. 检查示例代码中的注释
