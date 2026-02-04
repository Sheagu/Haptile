#!/bin/bash
# 批量导出所有数据集的视频 - Bash版本
# Batch export all datasets as videos

# 默认参数
DATA_ROOT="${1:-./shared/data/bc_data/yongqiang}"
OUTPUT_DIR="${2:-./videos}"
FPS="${3:-10}"

echo "=========================================="
echo "Batch Video Export (Shell Version)"
echo "=========================================="
echo "Data root: $DATA_ROOT"
echo "Output dir: $OUTPUT_DIR"
echo "FPS: $FPS"
echo "=========================================="
echo ""

# 检查数据目录是否存在
if [ ! -d "$DATA_ROOT" ]; then
    echo "❌ Error: Directory $DATA_ROOT does not exist"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 计数器
total=0
success=0
failed=0
skipped=0

# 遍历所有子目录
for dataset_dir in "$DATA_ROOT"/*/ ; do
    # 检查是否是目录
    if [ ! -d "$dataset_dir" ]; then
        continue
    fi
    
    # 检查是否包含pkl文件
    pkl_count=$(find "$dataset_dir" -maxdepth 1 -name "*.pkl" | wc -l)
    if [ "$pkl_count" -eq 0 ]; then
        continue
    fi
    
    total=$((total + 1))
    
    # 获取数据集名称
    dataset_name=$(basename "$dataset_dir")
    output_path="$OUTPUT_DIR/${dataset_name}.mp4"
    
    echo "[$total] Processing: $dataset_name ($pkl_count frames)"
    
    # 检查视频是否已存在
    if [ -f "$output_path" ]; then
        echo "  ⏭️  Skipped (already exists): $output_path"
        skipped=$((skipped + 1))
        continue
    fi
    
    # 导出视频
    python visualize_data.py "$dataset_dir" \
        --export-video "$output_path" \
        --fps "$FPS" 2>&1 | grep -E "(Processing|Success|exported|Error)" || true
    
    # 检查是否成功
    if [ -f "$output_path" ]; then
        echo "  ✅ Success: $output_path"
        success=$((success + 1))
    else
        echo "  ❌ Failed"
        failed=$((failed + 1))
    fi
    
    echo ""
done

# 打印总结
echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total datasets: $total"
echo "✅ Success: $success"
echo "⏭️  Skipped: $skipped"
echo "❌ Failed: $failed"
echo ""
echo "Videos saved to: $(realpath $OUTPUT_DIR)"
echo "=========================================="
