# Tactile Camera Setup Guide

## 摄像头配置

本系统使用混合摄像头配置：
- **Base Camera**: Intel RealSense (深度相机)
- **Tactile Sensors**: OpenCV Webcam (普通USB摄像头)

## 硬件连接

### 查看可用摄像头

在Linux系统上查看所有连接的摄像头：
```bash
ls -la /dev/video*
```

你会看到类似输出：
```
/dev/video0
/dev/video2
/dev/video4
/dev/video6
```

### 确定摄像头ID

使用以下Python脚本测试摄像头：
```python
import cv2

# 测试不同的摄像头ID
for i in [0, 2, 4, 6, 8]:
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
        ret, frame = cap.read()
        if ret:
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cap.release()
    else:
        print(f"Camera {i} is NOT available")
```

## 运行配置

### 基本用法（不使用触觉传感器）

```bash
python run_env.py \
  --no-use-tactile \
  --save-data \
  --hz 50
```

### 使用触觉传感器（默认启用）

```bash
python run_env.py \
  --use-tactile \
  --tactile-left-camera-id 2 \
  --tactile-right-camera-id 4 \
  --save-tactile-png \
  --save-data \
  --hz 50
```

或者简写（因为默认就是启用的）：
```bash
python run_env.py \
  --tactile-left-camera-id 22 \
  --tactile-right-camera-id 24 \
  --save-data
```

### 自定义分辨率

```bash
python launch_nodes.py \
  --realsense-width 1280 \
  --realsense-height 720 \
  --realsense-fps 30

python run_env.py \
  --realsense-width 1280 \
  --realsense-height 720 \
  --realsense-fps 30 \
  --tactile-width 640 \
  --tactile-height 480 \
  --tactile-left-camera-id 2 \
  --tactile-right-camera-id 4
```

说明：如果保持默认 `use_camera_node=True`，实际 RealSense 采集参数由 `launch_nodes.py` 决定；`run_env.py` 里的同名参数需要与之保持一致，避免保存数据时分辨率认知不一致。

### 双 RealSense 配置（第一视角 + 第三视角）

先列出 RealSense 设备 serial：
```bash
python3 realsense_id.py
```

再把两台设备都交给 camera node：
```bash
python launch_nodes.py \
  --cam-names 239122301234 128422270519 \
  --realsense-width 1280 \
  --realsense-height 720 \
  --realsense-fps 30

python run_env.py \
  --realsense-width 1280 \
  --realsense-height 720 \
  --realsense-fps 30 \
  --save-data \
  --save-png
```

说明：
- 第 0 路通常放第一视角，第 1 路放第三视角，顺序由 `--cam-names` 的顺序决定。
- 保存的 PKL 中，`base_camera_rgb` 形状会变成 `(2, H, W, 3)`。
- 保存 PNG 时会得到 `*-base_0.png` 和 `*-base_1.png`。

### 完整参数列表

```bash
python run_env.py \
  --robot-port 6000 \
  --hostname "127.0.0.1" \
  --hz 50 \
  --agent "quest" \
  --robot-type "ur5" \
  --save-data \
  --save-depth \
  --save-png \
  --use-tactile \
  --save-tactile-png \
  --realsense-width 640 \
  --realsense-height 480 \
  --realsense-fps 30 \
  --tactile-width 640 \
  --tactile-height 480 \
  --tactile-left-camera-id 2 \
  --tactile-right-camera-id 4 \
  --data-dir "./shared/data/bc_data"
```

### 禁用选项

```bash
# 禁用触觉传感器
python run_env.py --no-use-tactile

# 禁用保存PNG
python run_env.py --no-save-png --no-save-tactile-png

# 禁用摄像头显示
python run_env.py --no-show-camera-view
```

## 数据保存格式

### PKL文件内容
每个时间步保存的数据包含：
```python
{
    'base_camera_rgb': np.ndarray,      # (num_cameras, H, W, 3) 或 (H, W, 3)
    'base_camera_depth': np.ndarray,    # (num_cameras, H, W) 或 (H, W)
    'tactile_left_rgb': np.ndarray,     # (H, W, 3) - 如果启用
    'tactile_right_rgb': np.ndarray,    # (H, W, 3) - 如果启用
    'joint_positions': np.ndarray,
    'ee_pos_quat': np.ndarray,
    'control': np.ndarray,              # 动作指令
    'activated': bool
}
```

### PNG文件
如果启用PNG保存，会生成：
- `<timestamp>-base_0.png` - RealSense摄像头0的RGB图像
- `<timestamp>-base_1.png` - RealSense摄像头1的RGB图像 (如果有多个)
- `<timestamp>-tactile_left.png` - 左侧触觉传感器图像
- `<timestamp>-tactile_right.png` - 右侧触觉传感器图像

## 故障排除

### 问题：摄像头无法打开
```
RuntimeError: Failed to open camera X
```
**解决方案**：
1. 检查摄像头是否正确连接
2. 确认摄像头ID正确：`ls /dev/video*`
3. 检查权限：`sudo usermod -a -G video $USER`（重新登录后生效）

### 问题：RealSense无法初始化
```
RuntimeError: No RealSense devices found
```
**解决方案**：
1. 确认RealSense驱动已安装：`rs-enumerate-devices`
2. 重新插拔RealSense设备
3. 检查USB连接是否稳定（建议使用USB 3.0接口）

### 问题：帧率太低
**解决方案**：
1. 降低摄像头分辨率
2. 减少同时使用的摄像头数量
3. 禁用不需要的图像保存：`--save_png False --save_tactile_png False`

### 问题：触觉图像保存失败
**解决方案**：
1. 确认`use_tactile=True`
2. 检查obs字典中是否包含`tactile_left_rgb`和`tactile_right_rgb`键
3. 查看console输出的错误信息

## 性能优化建议

1. **降低分辨率**: 如果不需要高分辨率，可以使用较低的分辨率以提高帧率
   ```bash
   --realsense-width 320 --realsense-height 240
   --tactile-width 320 --tactile-height 240
   ```

2. **按需保存**: 只在需要时保存PNG图像
   ```bash
   --no-save-png --no-save-tactile-png
   ```

3. **关闭显示窗口**: 在生产环境中关闭实时显示
   ```bash
   --no-show-camera-view
   ```

4. **调整控制频率**: 根据实际需求调整
   ```bash
   --hz 30  # 降低到30Hz
   ```

## Tyro 参数说明

**重要**: tyro使用连字符(`-`)而不是下划线(`_`)作为命令行参数分隔符。

### 布尔参数
- 启用: `--use-tactile` 或省略（如果默认为True）
- 禁用: `--no-use-tactile`

### 示例对照表
| Python变量 | 命令行参数 |
|-----------|-----------|
| `use_tactile` | `--use-tactile` 或 `--no-use-tactile` |
| `save_data` | `--save-data` 或 `--no-save-data` |
| `tactile_left_camera_id` | `--tactile-left-camera-id 22` |
| `realsense_width` | `--realsense-width 640` |
