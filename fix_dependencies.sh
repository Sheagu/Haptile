#!/bin/bash
# 快速修复TeleUR可视化工具依赖问题

echo "=========================================="
echo "TeleUR Visualization Dependencies Fixer"
echo "=========================================="
echo ""

# 检查Python版本
echo "Checking Python version..."
python_version=$(python --version 2>&1)
echo "✓ $python_version"
echo ""

# 更新pip
echo "Updating pip..."
pip install --upgrade pip -q
echo "✓ pip updated"
echo ""

# 安装/更新核心依赖
echo "Installing/updating dependencies..."
pip install --upgrade numpy matplotlib opencv-python -q
echo "✓ numpy, matplotlib, opencv-python installed"
echo ""

# 检查tkinter
echo "Checking tkinter..."
python -c "import tkinter" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠ tkinter not found. Installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y python3-tk
        echo "✓ tkinter installed"
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3-tkinter
        echo "✓ tkinter installed"
    else
        echo "⚠ Please install tkinter manually for your system"
    fi
else
    echo "✓ tkinter already installed"
fi
echo ""

# 验证安装
echo "=========================================="
echo "Verification"
echo "=========================================="

errors=0

echo -n "numpy: "
python -c "import numpy; print(numpy.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Failed"
    errors=$((errors+1))
fi

echo -n "matplotlib: "
python -c "import matplotlib; print(matplotlib.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Failed"
    errors=$((errors+1))
fi

echo -n "opencv: "
python -c "import cv2; print(cv2.__version__)" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Failed"
    errors=$((errors+1))
fi

echo -n "tkinter: "
python -c "import tkinter; print('OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Failed"
    errors=$((errors+1))
fi

echo ""
echo "=========================================="

if [ $errors -eq 0 ]; then
    echo "✅ All dependencies installed successfully!"
    echo ""
    echo "You can now run:"
    echo "  python visualize_data.py <data_directory>"
else
    echo "⚠ $errors error(s) found. Please check the output above."
    echo ""
    echo "For more help, see TROUBLESHOOTING.md"
fi

echo "=========================================="
