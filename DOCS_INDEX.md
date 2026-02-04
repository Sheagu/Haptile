# TeleUR 项目文档总览

欢迎使用TeleUR遥操作机器人系统！本文档提供了快速导航指南。

## 📚 文档索引

### 🚀 快速开始（从这里开始！）
- **[QUICK_START.md](QUICK_START.md)** - 命令行使用快速指南，解决参数问题

### 📸 相机配置
- **[TACTILE_CAMERA_SETUP.md](TACTILE_CAMERA_SETUP.md)** - RealSense和触觉传感器配置详细指南

### 📊 数据可视化
- **[VISUALIZATION_README.md](VISUALIZATION_README.md)** - 可视化工具快速入门 ⭐推荐先看
- **[DATA_VISUALIZATION_GUIDE.md](DATA_VISUALIZATION_GUIDE.md)** - 完整的可视化和分析指南

### 🔧 故障排除
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 依赖安装和常见问题解决方案

## 🎯 常用工作流程

### 1️⃣ 数据采集
```bash
# 启动数据采集（带触觉传感器）
python run_env.py \
  --save-data \
  --tactile-left-camera-id 16 \
  --tactile-right-camera-id 24
```

**详细说明:** [QUICK_START.md](QUICK_START.md)

### 2️⃣ 数据查看
```bash
# 交互式查看数据
python visualize_data.py ./shared/data/bc_data/YOUR_FOLDER
```

**详细说明:** [VISUALIZATION_README.md](VISUALIZATION_README.md)

### 3️⃣ 数据分析
```bash
# 统计分析
python analyze_data.py ./shared/data/bc_data/YOUR_FOLDER
```

**详细说明:** [DATA_VISUALIZATION_GUIDE.md](DATA_VISUALIZATION_GUIDE.md)

### 4️⃣ 问题排查
```bash
# 自动修复依赖
./fix_dependencies.sh
```

**详细说明:** [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## 🛠️ 核心工具

### 数据采集
- `run_env.py` - 主要数据采集脚本
- `camera_node.py` - 相机节点（ZMQ）
- `robot_node.py` - 机器人节点（ZMQ）

### 相机驱动
- `cameras/realsense_camera.py` - Intel RealSense驱动
- `cameras/opencv_camera.py` - OpenCV USB摄像头驱动

### Agent控制
- `agents/quest_agent.py` - Meta Quest控制器
- `agents/dp_agent.py` - Diffusion Policy agent

### 数据分析
- `visualize_data.py` - 交互式数据查看器 ⭐
- `analyze_data.py` - 统计分析工具
- `read_pkl.py` - 简单pkl文件读取工具

### 辅助工具
- `fix_dependencies.sh` - 自动修复依赖
- `test_visualization.sh` - 测试可视化工具

## 📋 快速参考

### 命令行参数格式
```bash
# ✅ 正确：使用连字符
--save-data
--tactile-left-camera-id 22

# ❌ 错误：使用下划线或True/False
--save_data True
--tactile_left_camera_id 22
```

### 布尔参数
```bash
# 启用
--save-data

# 禁用  
--no-save-data
```

### 查看帮助
```bash
python run_env.py --help
python visualize_data.py --help
python analyze_data.py --help
```

## 🆘 遇到问题？

### 常见问题速查

| 问题 | 解决方案 | 详细文档 |
|-----|---------|---------|
| 参数解析错误 | 使用连字符`-`而非下划线`_` | [QUICK_START.md](QUICK_START.md) |
| 摄像头无法打开 | 检查设备ID和权限 | [TACTILE_CAMERA_SETUP.md](TACTILE_CAMERA_SETUP.md) |
| numpy._core错误 | 运行`./fix_dependencies.sh` | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| 无法显示图像 | 安装tkinter | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Quest控制速度快 | 调整`translation_scaling_factor` | [agents/quest_agent.py](agents/quest_agent.py) |

### 自动诊断
```bash
# 检查所有依赖
./fix_dependencies.sh

# 测试可视化工具
./test_visualization.sh ./shared/data/bc_data/YOUR_FOLDER
```

## 📦 环境要求

### 必需
- Python 3.8+
- numpy >= 1.21.0
- opencv-python >= 4.5.0
- matplotlib >= 3.3.0

### 可选
- pyrealsense2 (如果使用RealSense)
- tkinter (matplotlib GUI)

### 安装
```bash
# 基础依赖
pip install numpy matplotlib opencv-python

# RealSense支持
pip install pyrealsense2

# GUI支持 (Ubuntu/Debian)
sudo apt-get install python3-tk
```

## 🎓 学习路径

### 新手
1. 阅读 [QUICK_START.md](QUICK_START.md)
2. 运行第一个数据采集命令
3. 使用 [VISUALIZATION_README.md](VISUALIZATION_README.md) 查看数据

### 进阶
1. 配置多摄像头 [TACTILE_CAMERA_SETUP.md](TACTILE_CAMERA_SETUP.md)
2. 学习数据分析 [DATA_VISUALIZATION_GUIDE.md](DATA_VISUALIZATION_GUIDE.md)
3. 调整控制参数优化性能

### 专家
1. 修改agent行为 (`agents/`)
2. 自定义数据处理 (`learning/dp/`)
3. 扩展可视化功能 (`visualize_data.py`)

## 📊 数据格式

### 目录结构
```
shared/data/bc_data/
└── 0127_154230/              # 时间戳命名的会话
    ├── *.pkl                 # 每帧数据
    ├── *-tactile_left.png    # 触觉图像(可选)
    ├── *-tactile_right.png
    ├── *-base_0.png          # RealSense图像(可选)
    └── freq.txt              # 采集频率统计
```

### PKL文件内容
```python
{
    'base_camera_rgb': np.ndarray,      # RealSense RGB
    'base_camera_depth': np.ndarray,    # 深度图
    'tactile_left_rgb': np.ndarray,     # 左触觉
    'tactile_right_rgb': np.ndarray,    # 右触觉
    'joint_positions': np.ndarray,      # 关节角度
    'ee_pos_quat': np.ndarray,         # 末端位姿
    'control': np.ndarray,              # 控制指令
    'activated': bool                   # 激活状态
}
```

## 🔗 相关资源

### 代码仓库
- 主仓库: `/home/zhuo/git/teleUR`
- 数据目录: `./shared/data/bc_data/`
- 模型检查点: `./shared/ckpts/`

### 外部依赖
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [OpenCV](https://opencv.org/)
- [Matplotlib](https://matplotlib.org/)

## 💡 提示和技巧

1. **快速查看最新数据**
   ```bash
   alias view_latest='python visualize_data.py ./shared/data/bc_data/$(ls -t ./shared/data/bc_data/ | head -1)'
   ```

2. **批量分析**
   ```bash
   for dir in ./shared/data/bc_data/*/; do
       python analyze_data.py "$dir"
   done
   ```

3. **自动备份**
   ```bash
   cp -r ./shared/data/bc_data/ ./shared/data/bc_data_backup_$(date +%Y%m%d)
   ```

4. **清理旧数据**
   ```bash
   # 只保留最近10个会话
   ls -t ./shared/data/bc_data/ | tail -n +11 | xargs -I {} rm -rf ./shared/data/bc_data/{}
   ```

## 📞 获取支持

如果本文档无法解决你的问题：

1. 查看 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
2. 运行 `./fix_dependencies.sh`
3. 检查Python版本和依赖版本
4. 查看详细错误日志

---

**最后更新:** 2026-01-27  
**版本:** 1.0  
**维护者:** TeleUR Team
