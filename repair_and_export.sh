#!/bin/bash
# Script to fix corrupted pickle files and retry video export
# Usage: ./repair_and_export.sh <dataset_name>

set -e

# Configuration
DATA_DIR="./shared/data/bc_data"
OUTPUT_DIR="./shared/data/bc_data"
SCRIPT="visualize_data.py"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 0224_193531"
    exit 1
fi

DATASET=$1
DATASET_PATH="$DATA_DIR/$DATASET"

if [ ! -d "$DATASET_PATH" ]; then
    echo "Error: Dataset directory not found: $DATASET_PATH"
    exit 1
fi

echo "=========================================="
echo "Repairing dataset: $DATASET"
echo "=========================================="

# Activate conda environment
source activate py310

# Step 1: Scan for corrupted files
echo ""
echo "Step 1: Scanning for corrupted pickle files..."
echo "==========================================="
python3 check_and_clean_corrupted_pickle.py "$DATASET_PATH"

# Step 2: Ask user if they want to delete corrupted files
echo ""
read -p "Do you want to delete the corrupted files? (yes/no): " -r
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting corrupted pickle files..."
    python3 check_and_clean_corrupted_pickle.py "$DATASET_PATH" --delete
fi

# Step 3: Retry video export
echo ""
echo "Step 2: Attempting to export video..."
echo "======================================"
OUTPUT_VIDEO="$OUTPUT_DIR/${DATASET}.mp4"
python3 "$SCRIPT" "$DATASET_PATH" --export-video "$OUTPUT_VIDEO" --fps 10

echo ""
echo "=========================================="
echo "Done! Video saved to: $OUTPUT_VIDEO"
echo "=========================================="
