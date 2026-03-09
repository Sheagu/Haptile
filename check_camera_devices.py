#!/usr/bin/env python3
"""Diagnostic script to list all available camera devices and their v4l/by-path mappings."""

import os
import subprocess

def check_video_devices():
    """List all /dev/videoX devices and their by-path symlinks."""
    print("=" * 70)
    print("AVAILABLE V4L2 DEVICES")
    print("=" * 70)
    
    # Check /dev/video* devices
    video_devices = []
    for i in range(20):  # Check video0 to video19
        device_path = f"/dev/video{i}"
        if os.path.exists(device_path):
            video_devices.append(device_path)
    
    if not video_devices:
        print("No /dev/videoX devices found!")
        return
    
    print(f"\nFound {len(video_devices)} video device(s):")
    for device in video_devices:
        print(f"  - {device}")
    
    # Check v4l/by-path mappings
    print("\n" + "=" * 70)
    print("V4L/BY-PATH SYMLINKS")
    print("=" * 70)
    
    by_path_dir = "/dev/v4l/by-path"
    if not os.path.exists(by_path_dir):
        print(f"Directory {by_path_dir} does not exist!")
        return
    
    print(f"\nAvailable by-path symlinks:")
    by_path_devices = sorted(os.listdir(by_path_dir))
    
    if not by_path_devices:
        print("No by-path symlinks found!")
        return
    
    for symlink_name in by_path_devices:
        symlink_path = os.path.join(by_path_dir, symlink_name)
        if os.path.islink(symlink_path):
            target = os.path.realpath(symlink_path)
            print(f"  {symlink_path}")
            print(f"    -> {target}")
    
    # Try to get device information using v4l2-ctl
    print("\n" + "=" * 70)
    print("DEVICE DETAILS (via v4l2-ctl)")
    print("=" * 70)
    
    try:
        result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("v4l2-ctl not available or returned error")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("v4l2-ctl command not found. Install v4l-utils for more details:")
        print("  sudo apt install v4l-utils")

if __name__ == "__main__":
    check_video_devices()
    
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)
    print("""
To use these devices in launch_nodes.py, update CAM_PORTS:

    CAM_PORTS = {
        "435": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:3:1.0-video-index0",
    }

Or in run_env.py for tactile cameras:

    TACTILE_CAM_PORTS = {
        "left": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:2:1.0-video-index0",
        "right": "/dev/v4l/by-path/pci-0000:80:14.0-usb-0:3:1.0-video-index0",
    }
""")
