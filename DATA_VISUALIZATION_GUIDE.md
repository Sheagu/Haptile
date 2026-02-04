# 数据可视化和分析工具使用指南

## 工具概述

创建了两个数据分析工具：
1. **`visualize_data.py`** - 交互式可视化工具，查看图像和传感器数据
2. **`analyze_data.py`** - 统计分析工具，分析轨迹和数据分布

## 📊 工具1: visualize_data.py

### 功能
- 🖼️ 查看RGB图像（base camera和tactile sensors）
- 🌡️ 查看深度图
- 📈 显示关节位置、末端执行器姿态、控制命令
- ⏯️ 播放/暂停动画
- 🎬 导出为视频

### 基本使用

#### 1. 交互式查看数据
```bash
python visualize_data.py ./shared/data/bc_data/1225_143052
```

**交互功能：**
- 🎚️ **滑块**: 拖动查看不同帧
- ▶️ **Play/Pause**: 播放/暂停动画
- ⬅️ **Prev**: 上一帧
- ➡️ **Next**: 下一帧
- 键盘方向键也可以控制

#### 2. 导出为视频
```bash
python visualize_data.py ./shared/data/bc_data/1225_143052 \
  --export-video output.mp4 \
  --fps 10
```

参数说明：
- `--export-video`: 输出视频路径
- `--fps`: 视频帧率（默认10）

### 显示内容

可视化工具会自动检测并显示：
- ✅ Base Camera RGB（如果有多个RealSense，会显示多个）
- ✅ Base Camera Depth（深度图，使用jet colormap）
- ✅ Tactile Left（左侧触觉传感器）
- ✅ Tactile Right（右侧触觉传感器）
- ✅ 右侧信息面板：
  - 当前帧号
  - 文件名
  - 关节位置
  - 控制指令
  - 末端执行器位姿

## 📈 工具2: analyze_data.py

### 功能
- 📊 统计分析（均值、方差、最小值、最大值）
- 📉 绘制轨迹图
- 💾 保存分析结果

### 基本使用

```bash
python analyze_data.py ./shared/data/bc_data/1225_143052
```

### 输出内容

1. **终端输出：**
   ```
   ================================================
   Dataset Analysis: 1225_143052
   ================================================
   
   Total frames: 150
   
   Joint Positions Shape: (150, 7)
     Mean: [...]
     Std:  [...]
     Min:  [...]
     Max:  [...]
   
   End-Effector Poses Shape: (150, 6)
     Position:
       Mean: [...]
       Std:  [...]
   ```

2. **生成图表：** `trajectory_analysis.png`
   - 关节位置随时间变化
   - 末端执行器位置轨迹
   - 控制指令变化

## 🎯 实际使用示例

### 示例1: 快速查看刚采集的数据
```bash
# 假设你刚刚采集了数据到这个目录
python visualize_data.py ./shared/data/bc_data/0127_154230
```

### 示例2: 分析多个数据集
```bash
# 分析所有数据集
for dir in ./shared/data/bc_data/*/; do
    echo "Analyzing $dir"
    python analyze_data.py "$dir"
done
```

### 示例3: 创建演示视频
```bash
# 为每个数据集创建视频
python visualize_data.py ./shared/data/bc_data/0127_154230 \
  --export-video demos/demo_0127.mp4 \
  --fps 15
```

### 示例4: 检查数据质量
```bash
# 1. 先可视化检查图像质量
python visualize_data.py ./shared/data/bc_data/0127_154230

# 2. 分析统计信息
python analyze_data.py ./shared/data/bc_data/0127_154230
```

## 📁 数据目录结构

```
shared/data/bc_data/
├── 0127_154230/                    # 单次采集的数据
│   ├── 2026-01-27T15-42-30-123456.pkl
│   ├── 2026-01-27T15-42-30-156789.pkl
│   ├── ...
│   ├── freq.txt                    # 采集频率统计
│   └── trajectory_analysis.png     # 分析生成的图表（运行analyze_data.py后）
├── 0127_160315/
│   └── ...
└── ...
```

## 🔍 查看单个PKL文件内容

如果你想查看单个pkl文件的详细内容，可以使用Python：

```python
import pickle
import numpy as np

# 加载数据
with open("path/to/file.pkl", "rb") as f:
    data = pickle.load(f)

# 查看所有键
print("Keys:", data.keys())

# 查看具体数据
print("Joint positions:", data["joint_positions"])
print("Base RGB shape:", data["base_camera_rgb"].shape)
if "tactile_left_rgb" in data:
    print("Tactile left shape:", data["tactile_left_rgb"].shape)
```

或者使用命令行：
```bash
python -c "import pickle; import sys; data=pickle.load(open(sys.argv[1],'rb')); print('Keys:', list(data.keys())); print('Shapes:', {k:v.shape if hasattr(v,'shape') else type(v) for k,v in data.items()})" your_file.pkl
```

## 🐛 常见问题

### 问题1: matplotlib显示问题
```
UserWarning: Matplotlib is currently using agg, which is a non-GUI backend
```
**解决方案：**
```bash
# 安装tkinter
sudo apt-get install python3-tk
```

### 问题2: 视频导出失败
```
OpenCV: FFMPEG: tag 0x5634504d/'MP4V' is not supported
```
**解决方案：** 更改视频编码器或安装ffmpeg
```bash
sudo apt-get install ffmpeg
```

### 问题3: 内存不足
如果数据集很大，可能会内存不足。

**解决方案：** 分批处理或只加载部分数据
```python
# 修改visualize_data.py，只加载每N帧
self.pkl_files = sorted(glob.glob(str(self.data_dir / "*.pkl")))[::5]  # 每5帧取1帧
```

## 💡 高级用法

### 自定义可视化
你可以修改`visualize_data.py`来添加更多功能：

```python
# 在update函数中添加自定义显示
def update(val):
    # ...现有代码...
    
    # 添加速度计算
    if frame_idx > 0:
        prev_data = self.load_frame(frame_idx - 1)
        velocity = (data["joint_positions"] - prev_data["joint_positions"]) * fps
        info_str += f"Joint Velocity:\n{velocity}\n"
```

### 批量导出视频
```bash
#!/bin/bash
# export_all_videos.sh
for dir in ./shared/data/bc_data/*/; do
    dirname=$(basename "$dir")
    python visualize_data.py "$dir" \
      --export-video "videos/${dirname}.mp4" \
      --fps 10
done
```

## 📚 相关文件

- `read_pkl.py` - 原有的简单数据读取工具
- `run_env.py` - 数据采集脚本
- `QUICK_START.md` - 数据采集快速指南
- `TACTILE_CAMERA_SETUP.md` - 相机设置详细指南
