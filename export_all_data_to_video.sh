#!/bin/bash

# Export all data folders in bc_data to videos
# Usage:
#   ./export_all_data_to_video.sh                    # Export all datasets, FPS 10
#   ./export_all_data_to_video.sh --fps 20          # Export all with FPS 20
#   ./export_all_data_to_video.sh --output-dir ./videos  # Export to custom dir
#   ./export_all_data_to_video.sh --single 0224_184609   # Export single dataset

# Do NOT exit on error - we want to continue processing
# set -e

# Configuration
DATA_DIR="./shared/data/huawei_0310_2"
OUTPUT_DIR="./shared/data/huawei_0310_2"  # Default: same as input
FPS=10
SCRIPT="visualize_data.py"
SINGLE_DATASET=""
VERBOSE=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --fps)
            FPS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --single)
            SINGLE_DATASET="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --fps FPS              Video FPS (default: 10)"
            echo "  --output-dir DIR       Output directory for videos (default: same as input)"
            echo "  --single DATASET       Export only single dataset by name (e.g., 0224_184609)"
            echo "  --verbose              Show detailed output"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Export all with FPS 10"
            echo "  $0 --fps 20                                # Export all with FPS 20"
            echo "  $0 --single 0224_184609 --fps 30           # Export single dataset"
            echo "  $0 --output-dir ./videos --fps 15          # Export to custom directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory '$DATA_DIR' not found${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if Python script exists
if [ ! -f "$SCRIPT" ]; then
    echo -e "${RED}Error: Script '$SCRIPT' not found${NC}"
    exit 1
fi

# Function to export a single dataset
export_dataset() {
    local dataset_path="$1"
    local dataset_name=$(basename "$dataset_path")
    local output_video="${OUTPUT_DIR}/${dataset_name}.mp4"
    
    # Skip if not a directory
    if [ ! -d "$dataset_path" ]; then
        return 0
    fi
    
    # Skip if directory is empty
    if [ -z "$(ls -A "$dataset_path" 2>/dev/null)" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $dataset_name (empty directory)"
        return 0
    fi
    
    # Skip if video already exists
    if [ -f "$output_video" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $dataset_name (video already exists)"
        ((SKIPPED_COUNT++))
        return 0
    fi
    
    echo -e "${BLUE}[EXPORT]${NC} $dataset_name -> $output_video (FPS: $FPS)"
    
    if [ "$VERBOSE" = true ]; then
        echo "  Command: python $SCRIPT \"$dataset_path\" --export-video \"$output_video\" --fps $FPS"
    fi
    
    # Run the export command with error handling
    # Redirect errors to temporary file to check for failure
    local temp_error=$(mktemp)
    if python "$SCRIPT" "$dataset_path" --export-video "$output_video" --fps "$FPS" 2>"$temp_error" | tail -3; then
        # Check if there were actual errors
        if grep -q "Error\|Traceback\|Exception" "$temp_error" 2>/dev/null; then
            echo -e "${RED}[ERROR]${NC} $dataset_name failed:"
            cat "$temp_error"
            rm -f "$temp_error"
            return 1
        else
            echo -e "${GREEN}[OK]${NC} $dataset_name"
            echo ""
            rm -f "$temp_error"
            return 0
        fi
    else
        # Command failed
        echo -e "${RED}[ERROR]${NC} Failed to export $dataset_name"
        echo "Error details:"
        cat "$temp_error" | tail -10
        rm -f "$temp_error"
        return 1
    fi
}

# Main export logic
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Data Export to Video${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Data directory:  $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "FPS:             $FPS"
echo ""

EXPORTED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

if [ -n "$SINGLE_DATASET" ]; then
    # Export single dataset
    dataset_path="${DATA_DIR}/${SINGLE_DATASET}"
    if [ -d "$dataset_path" ]; then
        if export_dataset "$dataset_path"; then
            ((EXPORTED_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
    else
        echo -e "${RED}Error: Dataset '$SINGLE_DATASET' not found${NC}"
        exit 1
    fi
else
    # Export all datasets
    # Use simple directory listing instead of mapfile
    echo "Scanning datasets..."
    
    # Count total directories (don't cd, just use find)
    total=0
    while IFS= read -r -d '' dir; do
        ((total++))
    done < <(find "$DATA_DIR" -maxdepth 1 -type d ! -name "bc_data" -print0)
    
    if [ "$total" -eq 0 ]; then
        echo -e "${YELLOW}No datasets found in $DATA_DIR${NC}"
        exit 0
    fi
    
    echo -e "Found ${BLUE}$total${NC} dataset(s) to process."
    echo ""
    
    current=0
    # Use find without cd, so paths remain correct
    while IFS= read -r dataset_path; do
        if [ ! -d "$dataset_path" ]; then
            continue
        fi
        
        ((current++))
        
        echo -n "[$current/$total] "
        
        if export_dataset "$dataset_path"; then
            ((EXPORTED_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
    done < <(find "$DATA_DIR" -maxdepth 1 -type d ! -name "bc_data" | sort)
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Export Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Exported: ${GREEN}$EXPORTED_COUNT${NC}"
echo -e "Skipped:  ${YELLOW}$SKIPPED_COUNT${NC}"
echo -e "Failed:   ${RED}$FAILED_COUNT${NC}"
echo ""

if [ "$FAILED_COUNT" -gt 0 ]; then
    echo -e "${RED}Some exports failed. Please check the output above.${NC}"
    exit 1
else
    echo -e "${GREEN}Export completed successfully!${NC}"
    echo "Videos are saved in: $OUTPUT_DIR"
    exit 0
fi
            echo "Options:"
            echo "  --fps FPS              Video FPS (default: 10)"
            echo "  --output-dir DIR       Output directory for videos (default: same as input)"
            echo "  --single DATASET       Export only single dataset by name (e.g., 0224_184609)"
            echo "  --verbose              Show detailed output"
            echo "  --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                          # Export all with FPS 10"
            echo "  $0 --fps 20                                # Export all with FPS 20"
            echo "  $0 --single 0224_184609 --fps 30           # Export single dataset"
            echo "  $0 --output-dir ./videos --fps 15          # Export to custom directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo -e "${RED}Error: Data directory '$DATA_DIR' not found${NC}"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Check if Python script exists
if [ ! -f "$SCRIPT" ]; then
    echo -e "${RED}Error: Script '$SCRIPT' not found${NC}"
    exit 1
fi

# Function to export a single dataset
export_dataset() {
    local dataset_path="$1"
    local dataset_name=$(basename "$dataset_path")
    local output_video="${OUTPUT_DIR}/${dataset_name}.mp4"
    
    # Skip if not a directory
    if [ ! -d "$dataset_path" ]; then
        return 0
    fi
    
    # Skip if directory is empty
    if [ -z "$(ls -A "$dataset_path")" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $dataset_name (empty directory)"
        return 0
    fi
    
    # Skip if video already exists
    if [ -f "$output_video" ]; then
        echo -e "${YELLOW}[SKIP]${NC} $dataset_name (video already exists: $output_video)"
        return 0
    fi
    
    echo -e "${BLUE}[EXPORT]${NC} $dataset_name -> $output_video (FPS: $FPS)"
    
    if [ "$VERBOSE" = true ]; then
        echo "  Command: python $SCRIPT $dataset_path --export-video $output_video --fps $FPS"
    fi
    
    # Run the export command
    if python "$SCRIPT" "$dataset_path" --export-video "$output_video" --fps "$FPS" 2>&1 | tail -5; then
        echo -e "${GREEN}[OK]${NC} $dataset_name"
        echo ""
        return 0
    else
        echo -e "${RED}[ERROR]${NC} Failed to export $dataset_name"
        return 1
    fi
}

# Main export logic
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Data Export to Video${NC}"
echo -e "${BLUE}========================================${NC}"
echo "Data directory:  $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "FPS:             $FPS"
echo ""

EXPORTED_COUNT=0
SKIPPED_COUNT=0
FAILED_COUNT=0

if [ -n "$SINGLE_DATASET" ]; then
    # Export single dataset
    dataset_path="${DATA_DIR}/${SINGLE_DATASET}"
    if [ -d "$dataset_path" ]; then
        if export_dataset "$dataset_path"; then
            ((EXPORTED_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
    else
        echo -e "${RED}Error: Dataset '$SINGLE_DATASET' not found${NC}"
        exit 1
    fi
else
    # Export all datasets
    # Get sorted list of directories
    mapfile -t datasets < <(find "$DATA_DIR" -maxdepth 1 -type d ! -name "bc_data" -printf '%f\n' | sort)
    total=${#datasets[@]}
    
    if [ "$total" -eq 0 ]; then
        echo -e "${YELLOW}No datasets found in $DATA_DIR${NC}"
        exit 0
    fi
    
    echo -e "Found ${BLUE}$total${NC} dataset(s) to process."
    echo ""
    
    for ((i = 0; i < total; i++)); do
        dataset_name="${datasets[$i]}"
        
        if [ -z "$dataset_name" ]; then
            continue
        fi
        
        current=$((i + 1))
        dataset_path="${DATA_DIR}/${dataset_name}"
        
        echo -n "[$current/$total] "
        
        if export_dataset "$dataset_path"; then
            ((EXPORTED_COUNT++))
        else
            ((FAILED_COUNT++))
        fi
    done
fi

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Export Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "Exported: ${GREEN}$EXPORTED_COUNT${NC}"
echo -e "Skipped:  ${YELLOW}$SKIPPED_COUNT${NC}"
echo -e "Failed:   ${RED}$FAILED_COUNT${NC}"
echo ""

if [ "$FAILED_COUNT" -gt 0 ]; then
    echo -e "${RED}Some exports failed. Please check the output above.${NC}"
    exit 1
else
    echo -e "${GREEN}Export completed successfully!${NC}"
    echo "Videos are saved in: $OUTPUT_DIR"
    exit 0
fi
