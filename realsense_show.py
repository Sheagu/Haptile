import math
import time

import cv2
import numpy as np
import pyrealsense2 as rs

TILE_W, TILE_H = 640, 480

ctx = rs.context()
devices = ctx.query_devices()

if len(devices) == 0:
    print("No RealSense devices found.")
    exit(1)

n = len(devices)
print(f"Found {n} RealSense device(s).")

pipelines = []
for dev in devices:
    serial = dev.get_info(rs.camera_info.serial_number)
    name = dev.get_info(rs.camera_info.name)
    print(f"  {name} | serial: {serial}")

    pipeline = rs.pipeline(ctx)
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, TILE_W, TILE_H, rs.format.bgr8, 30)
    pipeline.start(config)
    pipelines.append((pipeline, serial))

# Per-camera state: last frame and FPS tracker
frames_cache = [np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)] * n
fps_trackers = [{"last": time.time(), "fps": 0.0, "count": 0} for _ in range(n)]

cols = math.ceil(math.sqrt(n))
rows = math.ceil(n / cols)

try:
    while True:
        for i, (pipeline, serial) in enumerate(pipelines):
            fs = pipeline.poll_for_frames()
            if not fs:
                continue
            color_frame = fs.get_color_frame()
            if not color_frame:
                continue

            img = np.asanyarray(color_frame.get_data())

            # Update FPS
            t = fps_trackers[i]
            t["count"] += 1
            now = time.time()
            elapsed = now - t["last"]
            if elapsed >= 0.5:
                t["fps"] = t["count"] / elapsed
                t["count"] = 0
                t["last"] = now

            # Overlay FPS and serial
            label = f"{serial}  {t['fps']:.1f} FPS"
            cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2, cv2.LINE_AA)

            frames_cache[i] = img

        # Build grid
        tiles = []
        for r in range(rows):
            row_imgs = []
            for c in range(cols):
                idx = r * cols + c
                if idx < n:
                    row_imgs.append(frames_cache[idx])
                else:
                    row_imgs.append(np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8))
            tiles.append(np.hstack(row_imgs))
        combined = np.vstack(tiles)

        cv2.imshow("RealSense Cameras", combined)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    for pipeline, _ in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()
