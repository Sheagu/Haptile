# 快速使用指南

## ✅ 正确的命令行用法

### 你的场景：使用触觉传感器（摄像头ID: 22和24）

```bash
python run_env.py \
  --tactile-left-camera-id 22 \
  --tactile-right-camera-id 24 \
  --save-data
```

注意：
- ✅ 使用 `--save-data` (连字符分隔)
- ❌ 不要用 `--save_data True` (下划线 + True)
- ✅ 使用 `--tactile-left-camera-id 22`
- ❌ 不要用 `--tactile_left_camera_id 22`

## 常用命令

### 1. 基本数据收集（带触觉）
```bash
python run_env.py --save-data --tactile-left-camera-id 22 --tactile-right-camera-id 24
```

### 2. 不使用触觉传感器
```bash
python run_env.py --no-use-tactile --save-data
```

### 3. 保存所有图像为PNG
```bash
python run_env.py --save-data --save-png --save-tactile-png
```

### 4. 自定义分辨率
```bash
python run_env.py \
  --realsense-width 1280 \
  --realsense-height 720 \
  --tactile-width 640 \
  --tactile-height 480
```

### 5. 调整控制频率
```bash
python run_env.py --hz 30 --save-data
```

### 6. 静默模式（不显示相机窗口）
```bash
python run_env.py --no-show-camera-view --save-data
```

## 布尔参数速查表

| 功能 | 启用 | 禁用 |
|-----|------|------|
| 触觉传感器 | `--use-tactile` | `--no-use-tactile` |
| 保存数据 | `--save-data` | `--no-save-data` |
| 保存深度 | `--save-depth` | `--no-save-depth` |
| 保存PNG | `--save-png` | `--no-save-png` |
| 保存触觉PNG | `--save-tactile-png` | `--no-save-tactile-png` |
| 显示摄像头 | `--show-camera-view` | `--no-show-camera-view` |

## 查看所有参数
```bash
python run_env.py --help
```

## 常见错误

### ❌ 错误示例
```bash
# 错误：使用下划线
python run_env.py --save_data True

# 错误：布尔值写True/False
python run_env.py --save-data True
```

### ✅ 正确示例
```bash
# 正确：使用连字符
python run_env.py --save-data

# 正确：禁用用no-前缀
python run_env.py --no-save-data
```

## 数据可视化

采集数据后，使用可视化工具查看：

### 查看数据（交互式）
```bash
# 查看指定文件夹的数据
python visualize_data.py ./shared/data/bc_data/YOUR_FOLDER

# 或查看最新采集的数据
python visualize_data.py ./shared/data/bc_data/$(ls -t ./shared/data/bc_data/ | head -1)
```

### 分析数据（统计信息）
```bash
python analyze_data.py ./shared/data/bc_data/YOUR_FOLDER
```

### 导出视频
```bash
python visualize_data.py ./shared/data/bc_data/YOUR_FOLDER --export-video output.mp4 --fps 10
```

详细说明请查看：
- `VISUALIZATION_README.md` - 可视化工具快速指南
- `DATA_VISUALIZATION_GUIDE.md` - 完整使用文档

## 故障排除

### 问题: 可视化工具报错 `No module named 'numpy._core'`

**快速修复：**
```bash
# 运行自动修复脚本
./fix_dependencies.sh
```

**手动修复：**
```bash
# 更新numpy
pip install --upgrade numpy

# 安装matplotlib和opencv
pip install matplotlib opencv-python

# 安装tkinter (Ubuntu/Debian)
sudo apt-get install python3-tk
```

更多故障排除信息请查看 `TROUBLESHOOTING.md`
