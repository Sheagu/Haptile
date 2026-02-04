# 批量导出视频工具使用指南

## 🎬 工具说明

创建了两个批量导出工具：

1. **`export_all_videos.py`** - Python版本（功能更丰富，推荐）
2. **`export_all_videos.sh`** - Shell脚本版本（简单快速）

## 🚀 快速开始

### 方法1: Python版本（推荐）

#### 导出所有数据集
```bash
python export_all_videos.py ./shared/data/bc_data
```

#### 导出到指定目录
```bash
python export_all_videos.py ./shared/data/bc_data --output-dir my_videos
```

#### 只导出最近5个数据集
```bash
python export_all_videos.py ./shared/data/bc_data --recent 5
```

#### 设置视频帧率
```bash
python export_all_videos.py ./shared/data/bc_data --fps 15
```

#### 强制重新导出（覆盖已存在的视频）
```bash
python export_all_videos.py ./shared/data/bc_data --no-skip-existing
```

#### 列出所有数据集
```bash
python export_all_videos.py ./shared/data/bc_data --list
```

### 方法2: Shell脚本版本

```bash
# 基本用法
./export_all_videos.sh

# 指定参数
./export_all_videos.sh <data_root> <output_dir> <fps>

# 例如
./export_all_videos.sh ./shared/data/bc_data ./videos 10
```

## 📋 完整参数说明

### Python版本参数

```bash
python export_all_videos.py --help
```

| 参数 | 说明 | 默认值 |
|-----|------|--------|
| `data_root` | 数据根目录路径 | 必需 |
| `--output-dir` | 输出视频目录 | `videos` |
| `--fps` | 视频帧率 | `10` |
| `--recent N` | 只导出最近N个数据集 | 导出全部 |
| `--list` | 列出所有数据集 | - |
| `--no-skip-existing` | 强制重新导出 | 跳过已存在 |

## 🎯 使用场景

### 场景1: 日常批量导出
```bash
# 每天工作结束后，批量导出当天的数据
python export_all_videos.py ./shared/data/bc_data --recent 10 --fps 10
```

### 场景2: 演示视频制作
```bash
# 制作高帧率演示视频
python export_all_videos.py ./shared/data/bc_data --fps 30 --output-dir demos
```

### 场景3: 数据质量检查
```bash
# 快速导出最新采集的数据
python export_all_videos.py ./shared/data/bc_data --recent 1 --fps 15
```

### 场景4: 完整归档
```bash
# 导出所有数据，不跳过已存在的（重新导出全部）
python export_all_videos.py ./shared/data/bc_data --no-skip-existing --output-dir archive
```

### 场景5: 查看数据集列表
```bash
# 先查看有哪些数据集
python export_all_videos.py ./shared/data/bc_data --list

# 输出示例：
# ==========================================================
# Datasets in ./shared/data/bc_data
# ==========================================================
# 
#   1. 0127_193334          |  324 frames | 2026-01-27 19:35:42
#   2. 0127_154230          |  150 frames | 2026-01-27 15:44:18
#   3. 0126_103521          |  280 frames | 2026-01-26 10:37:45
# 
# ==========================================================
# Total: 3 datasets
```

## 📊 输出示例

### 运行输出
```bash
$ python export_all_videos.py ./shared/data/bc_data --recent 3

============================================================
Export Recent 3 Datasets
============================================================
Data root: ./shared/data/bc_data
Output dir: videos
FPS: 10
============================================================

[1/3] Processing: 0127_193334
  Output: videos/0127_193334.mp4
Found 324 frames in ./shared/data/bc_data/0127_193334

=== Data Available ===
Base Camera RGB: True
Base Camera Depth: True
Tactile Left: True
Tactile Right: True
Joint Positions: True
End-Effector Pose: True
Control Commands: True

Exporting video to videos/0127_193334.mp4...
Processing frame 0/324
Processing frame 10/324
...
Video exported successfully to videos/0127_193334.mp4
  ✅ Success!

[2/3] Processing: 0127_154230
  ⏭️  Skipped (already exists)

[3/3] Processing: 0126_103521
  Output: videos/0126_103521.mp4
...

============================================================
Summary
============================================================
Total datasets: 3
✅ Success: 2
⏭️  Skipped: 1
❌ Failed: 0

Videos saved to: /home/zhuo/git/teleUR/videos
============================================================
```

## 🗂️ 目录结构

```
project/
├── shared/data/bc_data/          # 原始数据
│   ├── 0127_193334/              # 数据集1
│   │   ├── *.pkl
│   │   └── ...
│   ├── 0127_154230/              # 数据集2
│   │   ├── *.pkl
│   │   └── ...
│   └── ...
│
└── videos/                        # 导出的视频
    ├── 0127_193334.mp4           # 对应数据集1
    ├── 0127_154230.mp4           # 对应数据集2
    └── ...
```

## ⚙️ 高级用法

### 1. 自定义视频命名
修改`export_all_videos.py`中的输出路径：

```python
# 在第52行附近修改
output_path = output_dir / f"{dataset_name}.mp4"
# 改为
output_path = output_dir / f"demo_{dataset_name}_high_quality.mp4"
```

### 2. 并行处理（加速）
使用GNU Parallel或xargs并行处理：

```bash
# 获取所有数据集目录
find ./shared/data/bc_data -maxdepth 1 -type d -name "????_??????" | \
parallel -j 4 "python visualize_data.py {} --export-video videos/{/}.mp4 --fps 10"
```

### 3. 筛选特定时间段的数据
```bash
# 只导出1月27日的数据
python export_all_videos.py ./shared/data/bc_data
# 然后手动筛选或使用脚本
ls ./videos/0127_*.mp4
```

### 4. 集成到Makefile
```makefile
# Makefile
export-videos:
	python export_all_videos.py ./shared/data/bc_data --output-dir videos --fps 10

export-recent:
	python export_all_videos.py ./shared/data/bc_data --recent 5 --fps 15

list-datasets:
	python export_all_videos.py ./shared/data/bc_data --list

.PHONY: export-videos export-recent list-datasets
```

使用：
```bash
make export-videos
make export-recent
make list-datasets
```

## 🎨 视频格式说明

### 视频内容
- 多个摄像头画面水平拼接
- 包含：Base Camera RGB + Tactile Left + Tactile Right
- 左上角显示帧号

### 视频规格
- 编码格式: MP4V
- 分辨率: 根据图像自动调整
- 帧率: 可自定义（默认10 FPS）

## 💡 性能优化

### 提高导出速度
1. **降低帧率**: `--fps 5` 而不是 `--fps 30`
2. **减少处理数据**: `--recent 5` 只处理最近几个
3. **并行处理**: 使用GNU parallel或自行修改代码

### 减少存储空间
1. **降低帧率**: 更少的FPS = 更小的文件
2. **压缩设置**: 修改代码使用H.264编码
3. **只保留关键数据**: 只导出特定摄像头的画面

## 🐛 故障排除

### 问题1: 内存不足
**症状**: 处理大数据集时程序崩溃

**解决方案**:
- 使用 `--recent N` 分批处理
- 增加系统swap空间
- 修改代码每次只加载必要的数据

### 问题2: 视频无法播放
**症状**: 导出的视频文件无法打开

**解决方案**:
```bash
# 安装ffmpeg
sudo apt-get install ffmpeg

# 转换视频格式
ffmpeg -i input.mp4 -vcodec libx264 output_fixed.mp4
```

### 问题3: 导出速度慢
**症状**: 处理一个数据集需要很长时间

**解决方案**:
- 降低FPS
- 使用更快的存储设备
- 减少图像分辨率（修改代码）

### 问题4: 跳过了我想重新导出的视频
**症状**: 视频已存在但质量不好，想重新导出

**解决方案**:
```bash
# 使用 --no-skip-existing 强制重新导出
python export_all_videos.py ./shared/data/bc_data --no-skip-existing
```

## 📞 相关工具

- `visualize_data.py` - 单个数据集的交互式查看和导出
- `analyze_data.py` - 数据统计分析
- `VISUALIZATION_README.md` - 可视化工具总览

## 🔗 工作流集成

### 完整数据处理流程
```bash
# 1. 采集数据
python run_env.py --save-data

# 2. 快速查看最新数据
python visualize_data.py ./shared/data/bc_data/$(ls -t ./shared/data/bc_data/ | head -1)

# 3. 批量导出视频
python export_all_videos.py ./shared/data/bc_data --recent 1

# 4. 数据分析
python analyze_data.py ./shared/data/bc_data/$(ls -t ./shared/data/bc_data/ | head -1)
```

### 自动化脚本
创建 `daily_export.sh`:
```bash
#!/bin/bash
# 每日自动导出脚本

DATE=$(date +%Y%m%d)
python export_all_videos.py ./shared/data/bc_data \
  --output-dir videos/${DATE} \
  --fps 10

echo "Daily export completed: videos/${DATE}"
```

设置cron job:
```bash
# 每天晚上10点自动导出
0 22 * * * /path/to/daily_export.sh
```

## ✅ 总结

### Python版本优势
- ✅ 功能完整（列表、最近N个、跳过已存在）
- ✅ 详细的进度显示和错误处理
- ✅ 统计报告

### Shell版本优势  
- ✅ 简单快速，无需额外依赖
- ✅ 易于集成到现有脚本
- ✅ 资源占用少

### 推荐使用场景
- **日常使用**: Python版本
- **快速脚本**: Shell版本
- **自动化**: 两者皆可

现在你可以轻松地批量导出所有数据集的视频了！🎬
