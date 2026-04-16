# 📹 Batch Video Export Guide

This guide explains how to batch export videos from all datasets in a folder using the TeleUR video export tools.

## 🎯 Quick Start

### Export All Datasets

```bash
# Export all datasets to default 'videos' folder
python export_all_videos.py ./shared/data/bc_data

# Export to a custom directory with custom frame rate
python export_all_videos.py ./shared/data/bc_data --output-dir my_videos --fps 15
```

### Export Recent Datasets Only

```bash
# Export only the 5 most recent datasets
python export_all_videos.py ./shared/data/bc_data --recent 5

# Export the 10 most recent with custom settings
python export_all_videos.py ./shared/data/bc_data --recent 10 --output-dir recent_videos --fps 20
```

### List Datasets

```bash
# View all available datasets without exporting
python export_all_videos.py ./shared/data/bc_data --list
```

## 📋 Command Reference

### Python Script (`export_all_videos.py`)

```bash
python export_all_videos.py [DATA_ROOT] [OPTIONS]
```

**Required Arguments:**
- `DATA_ROOT`: Root directory containing dataset folders (e.g., `./shared/data/bc_data`)

**Optional Arguments:**
- `--output-dir DIR`: Output directory for videos (default: `videos`)
- `--fps N`: Video frame rate (default: 10)
- `--recent N`: Only export N most recent datasets
- `--list`: List all datasets and exit (no export)
- `--no-skip-existing`: Re-export even if video already exists

**Examples:**

```bash
# Export all datasets
python export_all_videos.py ./shared/data/bc_data

# Custom output directory and frame rate
python export_all_videos.py ./shared/data/bc_data --output-dir exports --fps 30

# Export only 3 most recent datasets
python export_all_videos.py ./shared/data/bc_data --recent 3

# Force re-export (overwrite existing videos)
python export_all_videos.py ./shared/data/bc_data --no-skip-existing

# List all available datasets
python export_all_videos.py ./shared/data/bc_data --list
```

### Shell Script (`export_all_videos.sh`)

```bash
./export_all_videos.sh [DATA_ROOT] [OUTPUT_DIR] [FPS]
```

**Arguments:**
- `DATA_ROOT`: Data root directory (default: `./shared/data/bc_data`)
- `OUTPUT_DIR`: Output video directory (default: `./videos`)
- `FPS`: Video frame rate (default: 10)

**Examples:**

```bash
# Use all defaults
./export_all_videos.sh

# Custom data root
./export_all_videos.sh ./my_data/recordings

# Custom output directory and FPS
./export_all_videos.sh ./shared/data/bc_data ./my_videos 15
```

## 🎬 Output Format

### Video Files

- **Format**: MP4 (H.264)
- **Filename**: `{dataset_name}.mp4`
- **Default Frame Rate**: 10 FPS
- **Resolution**: Same as input images

### Example Output

```
videos/
├── 0606_113104.mp4  (12.3 MB)
├── 0606_113127.mp4  (15.7 MB)
├── 0606_113206.mp4  (14.2 MB)
└── ...
```

## 🔧 Advanced Usage

### Selective Export

Export only datasets matching a pattern:

```bash
# Export datasets containing "task1" in name
for dir in ./shared/data/bc_data/*task1*/; do
    python visualize_data.py "$dir" --export-video "videos/$(basename $dir).mp4"
done
```

### Custom Frame Rate for Different Tasks

```bash
# Slow tasks - lower FPS
python export_all_videos.py ./data/slow_tasks --fps 5

# Fast tasks - higher FPS
python export_all_videos.py ./data/fast_tasks --fps 20
```

### Parallel Export (Advanced)

For large datasets, you can export multiple videos in parallel:

```bash
# Using GNU parallel (install: sudo apt install parallel)
find ./shared/data/bc_data -maxdepth 1 -type d -name "*_*" | \
    parallel -j 4 python visualize_data.py {} --export-video "videos/{/}.mp4"
```

### Export with Custom Naming

```bash
# Add prefix to all exported videos
python export_all_videos.py ./shared/data/bc_data --output-dir videos_demo

# Then rename with prefix
cd videos_demo
for file in *.mp4; do mv "$file" "demo_$file"; done
```

## 📊 Progress Tracking

The export tool provides detailed progress information:

```
==========================================================
Batch Video Export
==========================================================
Data root: ./shared/data/bc_data
Output dir: videos
Found 8 datasets
FPS: 10
Skip existing: True
==========================================================

[1/8] Processing: 0606_113104 (150 frames)
  Output: videos/0606_113104.mp4
  ✅ Success! (12.3 MB, 8.2s)

[2/8] Processing: 0606_113127 (200 frames)
  Output: videos/0606_113127.mp4
  ⏭️  Skipped (already exists, 15.7 MB)

[3/8] Processing: 0606_113206 (180 frames)
  Output: videos/0606_113206.mp4
  ✅ Success! (14.2 MB, 9.5s)

...

==========================================================
Summary
==========================================================
Total datasets: 8
✅ Success: 6
⏭️  Skipped: 2
❌ Failed: 0
⏱️  Duration: 45.3s

Videos saved to: /home/user/teleUR/videos
==========================================================
```

## 🐛 Troubleshooting

### Issue: "No dataset folders found"

**Solution:** Make sure your data directory contains folders with `.pkl` files:

```bash
# Check dataset structure
ls -la ./shared/data/bc_data/
# Should show folders like: 0606_113104/, 0606_113127/, etc.

# Check if folders contain pkl files
ls ./shared/data/bc_data/0606_113104/*.pkl
```

### Issue: Export fails with "No module named 'cv2'"

**Solution:** Install required dependencies:

```bash
pip install opencv-python numpy
# or
./fix_dependencies.sh
```

### Issue: Videos are too large

**Solution:** Reduce frame rate or compress output:

```bash
# Lower FPS
python export_all_videos.py ./shared/data/bc_data --fps 5

# Post-process with ffmpeg for better compression
ffmpeg -i input.mp4 -c:v libx264 -crf 23 -preset slow output_compressed.mp4

# Batch compress all videos
for video in videos/*.mp4; do
    ffmpeg -i "$video" -c:v libx264 -crf 23 -preset slow "compressed/$(basename $video)"
done
```

### Issue: Export is too slow

**Solution:**
1. Skip already exported videos (default behavior)
2. Reduce frame rate
3. Export only recent datasets
4. Use parallel export (see Advanced Usage)

```bash
# Fast export - only recent 5 datasets at 5 FPS
python export_all_videos.py ./shared/data/bc_data --recent 5 --fps 5
```

### Issue: Some exports fail

**Solution:** Check the error messages in the summary. Common issues:
- Corrupted pkl files: Re-collect the dataset
- Missing image files: Check if all frames have corresponding images
- Insufficient disk space: Free up space or use a different output directory

```bash
# Check disk space
df -h

# Check specific dataset
python visualize_data.py ./shared/data/bc_data/problematic_dataset --info
```

### Issue: Video playback is choppy

**Solution:** 
- Increase frame rate for smoother playback
- Ensure video player supports H.264 codec
- Re-export with higher quality settings

```bash
# Higher FPS for smoother playback
python export_all_videos.py ./shared/data/bc_data --fps 30
```

## 💡 Tips & Best Practices

### 1. **Regular Exports**

Export videos regularly to catch data collection issues early:

```bash
# After each recording session, export recent datasets
python export_all_videos.py ./shared/data/bc_data --recent 10
```

### 2. **Quality Control**

Review exported videos to verify:
- ✅ Proper camera alignment and lighting
- ✅ Smooth robot motion (no jerky movements)
- ✅ Complete task execution
- ✅ No frame drops or artifacts
- ✅ Correct gripper open/close states

### 3. **Organized Storage**

Keep exports organized by date or task:

```bash
# Export with dated directory
DATE=$(date +%Y%m%d)
python export_all_videos.py ./shared/data/bc_data --output-dir "exports/$DATE"

# Export by task type
python export_all_videos.py ./data/pick_tasks --output-dir "videos/pick"
python export_all_videos.py ./data/place_tasks --output-dir "videos/place"
```

### 4. **Disk Space Management**

Monitor disk usage and clean up old exports:

```bash
# Check video folder size
du -sh videos/

# List largest videos
du -sh videos/* | sort -hr | head -10

# Keep only recent exports (delete videos older than 30 days)
find videos/ -name "*.mp4" -mtime +30 -delete
```

### 5. **Frame Rate Guidelines**

Choose appropriate FPS based on task speed:

| Task Type | Recommended FPS | Use Case |
|-----------|----------------|----------|
| Slow manipulation | 5-10 FPS | Precise assembly, careful placement |
| Normal speed tasks | 10-15 FPS | General pick-and-place operations |
| Fast motion tasks | 20-30 FPS | Quick movements, throwing |
| High-speed analysis | 30-60 FPS | Detailed motion analysis |

### 6. **Backup Strategy**

Always keep original pkl data and export videos separately:

```
backups/
├── data/           # Original pkl files (CRITICAL - never delete)
│   └── bc_data/
│       ├── 0606_113104/
│       ├── 0606_113127/
│       └── ...
└── videos/         # Exported videos (can be regenerated)
    └── 2024/
        ├── 0606_113104.mp4
        ├── 0606_113127.mp4
        └── ...
```

### 7. **Incremental Export Workflow**

Efficient workflow for continuous data collection:

```bash
# Day 1: Initial export
python export_all_videos.py ./shared/data/bc_data --output-dir videos

# Day 2: Export only new datasets (skips existing)
python export_all_videos.py ./shared/data/bc_data --output-dir videos

# Day 3: Quick check of today's recordings
python export_all_videos.py ./shared/data/bc_data --recent 5 --output-dir videos
```

### 8. **Automation with Cron**

Schedule automatic exports (Linux/Mac):

```bash
# Add to crontab (crontab -e)
# Export every day at 6 PM
0 18 * * * cd /home/user/teleUR && python export_all_videos.py ./shared/data/bc_data

# Export recent datasets every hour
0 * * * * cd /home/user/teleUR && python export_all_videos.py ./shared/data/bc_data --recent 5
```

## 📚 Related Tools

- **`visualize_data.py`**: Interactive viewer and single video export
- **`visualize_data_all.py`**: Multi-dataset interactive viewer
- **`analyze_data.py`**: Statistical analysis and trajectory plotting
- **`read_pkl.py`**: Inspect individual dataset files

## 🔗 See Also

- [Data Visualization Guide](DATA_VISUALIZATION_GUIDE.md)
- [Quick Start Guide](QUICK_START.md)
- [Troubleshooting](TROUBLESHOOTING.md)
- [Documentation Index](DOCS_INDEX.md)

## 🎓 Tutorial: Complete Workflow

### Step 1: Check Available Datasets

```bash
python export_all_videos.py ./shared/data/bc_data --list
```

### Step 2: Export Recent Datasets for Quick Review

```bash
python export_all_videos.py ./shared/data/bc_data --recent 5 --fps 15
```

### Step 3: Review Videos

```bash
# Open video folder
xdg-open videos/  # Linux
# or
open videos/  # Mac
```

### Step 4: Export All Datasets for Archive

```bash
python export_all_videos.py ./shared/data/bc_data --output-dir archive --fps 10
```

### Step 5: Verify Results

```bash
# Check export statistics
ls -lh videos/ | wc -l  # Count videos
du -sh videos/          # Total size
```

---

**Need help?** See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.
