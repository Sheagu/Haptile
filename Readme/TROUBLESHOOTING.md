# 依赖安装和故障排除指南

## 🐛 常见错误及解决方案

### 错误1: `ModuleNotFoundError: No module named 'numpy._core'`

这是numpy版本兼容性问题（通常发生在numpy 1.x和2.x之间）。

**解决方案A: 更新numpy（推荐）**
```bash
pip install --upgrade numpy
```

**解决方案B: 降级numpy**
```bash
pip install "numpy<2.0"
```

**解决方案C: 已在代码中修复**
最新版本的`visualize_data.py`和`analyze_data.py`已经包含了兼容性修复，直接使用即可。

### 错误2: `ImportError: No module named 'tkinter'`

matplotlib需要tkinter用于GUI显示。

**解决方案:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3-tk

# Fedora/RHEL
sudo dnf install python3-tkinter

# macOS (通常已安装)
# 如果没有，通过homebrew安装: brew install python-tk
```

### 错误3: `ImportError: No module named 'cv2'`

缺少OpenCV。

**解决方案:**
```bash
pip install opencv-python
```

### 错误4: `ImportError: No module named 'matplotlib'`

缺少matplotlib。

**解决方案:**
```bash
pip install matplotlib
```

### 错误5: matplotlib警告 "using agg backend"

这意味着matplotlib无法使用GUI后端。

**解决方案:**
```bash
# 安装tkinter
sudo apt-get install python3-tk

# 或在代码中已修复（使用TkAgg backend）
```

## 📦 完整依赖安装

### 方法1: 使用pip安装所有依赖
```bash
pip install numpy matplotlib opencv-python
```

### 方法2: 从requirements.txt安装
如果项目根目录有requirements.txt：
```bash
pip install -r requirements.txt
```

### 方法3: 创建虚拟环境（推荐）
```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install numpy matplotlib opencv-python pyrealsense2
```

## 🔧 验证安装

运行以下命令验证所有依赖是否正确安装：

```bash
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import matplotlib; print('matplotlib:', matplotlib.__version__)"
python -c "import cv2; print('opencv:', cv2.__version__)"
python -c "import tkinter; print('tkinter: OK')"
```

预期输出：
```
numpy: 1.24.3 (或更高版本)
matplotlib: 3.7.1 (或更高版本)
opencv: 4.8.0 (或更高版本)
tkinter: OK
```

## 🚀 快速修复脚本

保存以下内容为 `fix_dependencies.sh`:

```bash
#!/bin/bash
echo "Fixing TeleUR visualization dependencies..."

# 更新pip
pip install --upgrade pip

# 安装/更新核心依赖
pip install --upgrade numpy matplotlib opencv-python

# 检查tkinter
python3 -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing tkinter..."
    sudo apt-get update
    sudo apt-get install -y python3-tk
fi

# 验证安装
echo -e "\n=== Verification ==="
python -c "import numpy; print('✓ numpy:', numpy.__version__)"
python -c "import matplotlib; print('✓ matplotlib:', matplotlib.__version__)"
python -c "import cv2; print('✓ opencv:', cv2.__version__)"
python -c "import tkinter; print('✓ tkinter: OK')"

echo -e "\n✅ All dependencies installed!"
```

运行：
```bash
chmod +x fix_dependencies.sh
./fix_dependencies.sh
```

## 🔍 具体错误诊断

### 如果遇到pickle加载错误

**症状:**
```
ModuleNotFoundError: No module named 'numpy._core'
或
AttributeError: module 'numpy' has no attribute '_core'
```

**原因:** 
- 数据是用numpy 2.x保存的，但你在用numpy 1.x加载
- 或反过来

**解决方案:**
1. **更新到最新版本（推荐）:**
   ```bash
   pip install --upgrade numpy
   ```

2. **使用兼容模式（已在代码中实现）:**
   最新的visualize_data.py已包含兼容性代码：
   ```python
   import numpy as np
   if not hasattr(np, '_core'):
       np._core = np.core
   ```

3. **重新保存数据（如果需要）:**
   如果问题持续，可能需要用统一的numpy版本重新采集数据。

## 📋 推荐的环境配置

### Python版本
- **推荐:** Python 3.8 - 3.11
- **最低:** Python 3.7

### 依赖版本
```
numpy>=1.21.0
matplotlib>=3.3.0
opencv-python>=4.5.0
```

### 完整requirements.txt示例
```txt
numpy>=1.21.0,<2.0
matplotlib>=3.3.0
opencv-python>=4.5.0
pyrealsense2>=2.50.0
tyro>=0.5.0
termcolor>=2.0.0
pynput>=1.7.0
```

## 💡 预防措施

### 1. 使用虚拟环境
始终在虚拟环境中工作，避免全局包冲突：
```bash
python -m venv teleUR_env
source teleUR_env/bin/activate
pip install -r requirements.txt
```

### 2. 固定版本
在requirements.txt中固定版本号：
```txt
numpy==1.24.3
matplotlib==3.7.1
opencv-python==4.8.0
```

### 3. 定期更新
保持依赖更新，但在更新前备份：
```bash
pip list --format=freeze > requirements_backup.txt
pip install --upgrade numpy matplotlib opencv-python
```

## 🆘 还是不行？

### 尝试完全重装
```bash
# 卸载所有相关包
pip uninstall numpy matplotlib opencv-python -y

# 清除pip缓存
pip cache purge

# 重新安装
pip install numpy matplotlib opencv-python
```

### 检查Python环境
```bash
# 查看Python路径
which python
python --version

# 查看已安装的包
pip list | grep -E "numpy|matplotlib|opencv"

# 查看包安装位置
python -c "import numpy; print(numpy.__file__)"
```

### 使用conda（如果pip不行）
```bash
conda create -n teleUR python=3.9
conda activate teleUR
conda install numpy matplotlib opencv
pip install pyrealsense2 tyro termcolor pynput
```

## 📞 获取帮助

如果以上方法都不能解决问题：

1. 运行诊断命令：
   ```bash
   python --version
   pip list
   python -c "import sys; print(sys.path)"
   ```

2. 查看完整错误信息
3. 检查是否有多个Python版本冲突
4. 考虑使用Docker容器隔离环境
