# 快速使用指南

## ✅ 正确的命令行用法
```commandline
python launch_nodes.py
```

### 你的场景：使用触觉传感器（摄像头ID: 22和24）

python launch_nodes.py 

#### 修改camera id读取方式为端口号：/dev/v4l/by-path/xxxx

```bash
python run_env.py \
  --tactile-left-camera-id 2 \
  --tactile-right-camera-id 12 \
  --save-data
```
python run_env.py \
  --tactile-left-camera-id 2 \
  --save-data

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

### 1.1 触觉图像自动裁剪并映射到矩形
```bash
python run_env.py \
  --save-data \
  --tactile-left-camera-id 22 \
  --tactile-right-camera-id 24
```

说明：
- 触觉相机默认强制执行 crop/warp，不需要再传额外参数
- 程序会优先查找 `robo_test/sensor_config_left.json` 和 `robo_test/sensor_config_right.json`
- 之后才回退到旧命名：`robo_test/tactile_left_sensor_config.json`、`robo_test/left_sensor_config.json`、`robo_test/sensor_config.json`
- 左右传感器会分别读取各自配置，不再默认共用同一组点
- 配置文件默认读取 `points` 字段，点顺序需要与 `marker_tracking.py` 一致：`[左上, 右上, 右下, 左下]`
- 如果 JSON 里包含 `tactile_left` 或 `tactile_right` 节点，会优先读取对应节点
- 如果没写 `output_size`，程序会按四点位置自动推断输出矩形尺寸

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
python launch_nodes.py \
  --realsense-width 1280 \
  --realsense-height 720

python run_env.py \
  --realsense-width 1280 \
  --realsense-height 720 \
  --tactile-width 640 \
  --tactile-height 480
```

说明：默认 `use_camera_node=True`，所以 RealSense 分辨率需要在 `launch_nodes.py` 和 `run_env.py` 两边保持一致；否则 `run_env.py` 里的 `--realsense-width/height` 不会改变实际采集视角。

### 4.1 加第二个 RealSense 作为第三视角
先查看两台 RealSense 的 serial：
```bash
python3 realsense_id.py
```

然后把第一视角和第三视角的 serial 一起传给 `launch_nodes.py`：
```bash
python launch_nodes.py \
  --cam-names 239122301234 128422270519 \
  --realsense-width 1280 \
  --realsense-height 720

python run_env.py \
  --realsense-width 1280 \
  --realsense-height 720 \
  --save-data \
  --save-png
```

说明：
- `base_camera_rgb[0]` 是第一台 RealSense，`base_camera_rgb[1]` 是第二台 RealSense。
- 如果启用 PNG 保存，会自动生成 `*-base_0.png` 和 `*-base_1.png`。
- `show_camera_view` 现在支持多 RealSense，会在同一个窗口里显示两路 RGB-D。

### 5. 调整控制频率
```bash
python run_env.py --hz 30 --save-data
```

### 6. 静默模式（不显示相机窗口）
```bash
python run_env.py --no-show-camera-view --save-data
```

### 7. 添加触觉反馈和重置功能
使用UDP通信实现触觉反馈，将触觉传感器里面的marker motion大小映射为headset手柄震动强度；同时设置右手柄按键B来重置触觉图像参考帧为当前帧。

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
