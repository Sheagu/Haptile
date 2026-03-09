# Fix for EOFError: Ran out of input in Video Export

## Problem
When exporting dataset `0224_193531` to video, the script fails with:
```
EOFError: Ran out of input
  File "visualize_data.py", line 80, in load_frame
    return pickle.load(f)
```

## Root Cause
One or more pickle files in the dataset are corrupted or truncated:
- **Affected Dataset**: `0224_193531`
- **Corrupted File**: `2026-02-24T19-36-06-474996.pkl`
- **Total Files**: 744, with 1 corrupted

This typically occurs when:
- Data collection was interrupted (power loss, process crash)
- File wasn't properly flushed to disk
- Disk I/O error during writing

## Solution

### Option 1: Quick Fix (Delete Corrupted File)

```bash
# Navigate to project directory
cd /home/shiyigu/Documents/project/teleUR

# Activate the Python environment
conda activate py310

# Delete the corrupted file
rm shared/data/bc_data/0224_193531/2026-02-24T19-36-06-474996.pkl

# Export the video (now it will skip the corrupted frame)
python3 visualize_data.py shared/data/bc_data/0224_193531 \
  --export-video shared/data/bc_data/0224_193531.mp4 --fps 10
```

### Option 2: Use the Repair Script (Automated)

```bash
conda activate py310
./repair_and_export.sh 0224_193531
```

This script will:
1. Scan for all corrupted pickle files
2. Ask for confirmation to delete them
3. Automatically retry the video export

## New Features Added

### 1. Robust Error Handling in `visualize_data.py`
- Automatically skips corrupted frames instead of crashing
- Finds the first valid frame for initialization
- Reports which frames were skipped during export

### 2. Utility Script: `check_and_clean_corrupted_pickle.py`

Check for corrupted files in a dataset:
```bash
python3 check_and_clean_corrupted_pickle.py shared/data/bc_data/0224_193531
```

With detailed errors:
```bash
python3 check_and_clean_corrupted_pickle.py shared/data/bc_data/0224_193531 -v
```

Delete corrupted files:
```bash
python3 check_and_clean_corrupted_pickle.py shared/data/bc_data/0224_193531 --delete
```

Scan multiple datasets recursively:
```bash
python3 check_and_clean_corrupted_pickle.py shared/data/bc_data -r
```

## Results

After applying the fix:
- **Frames Successfully Processed**: 743 out of 744
- **Input File**: `0224_193531` (744 frames originally)
- **Output Video**: 12MB MP4 video at 10 FPS
- **Status**: ✅ Successfully exported

## Prevention for Future

To prevent this issue:

1. **Check data validity** before exporting:
   ```bash
   python3 check_and_clean_corrupted_pickle.py <dataset_path>
   ```

2. **Monitor disk space** during data collection

3. **Ensure proper shutdown** of recording processes

4. **Regular backups** of important datasets

## Modified Files

1. **[visualize_data.py](visualize_data.py)**: Added error handling for corrupted frames
2. **[check_and_clean_corrupted_pickle.py](check_and_clean_corrupted_pickle.py)**: New utility to scan and clean corrupted files
3. **[repair_and_export.sh](repair_and_export.sh)**: New script for automated repair and export

## Technical Details

The fixes add two layers of protection:

1. **Initialization**: When loading the first frame to detect available data, it now tries multiple frames until finding a valid one
2. **Processing**: During frame iteration, any `EOFError` or `UnpicklingError` is caught, the frame is skipped, and processing continues

This allows the export to complete successfully even with corrupted frames, while preserving all valid data.
