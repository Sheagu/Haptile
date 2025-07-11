import pyrealsense2 as rs
import numpy as np
import cv2

# Configure depth, color, and motion streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.gyro)  # Enable gyroscope
config.enable_stream(rs.stream.accel) # Enable accelerometer

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent set of frames
        frames = pipeline.wait_for_frames()
        
        color_frame = frames.get_color_frame()
        gyro_frame = frames.first_or_default(rs.stream.gyro)
        accel_frame = frames.first_or_default(rs.stream.accel)
        
        # Process color frame
        if color_frame:
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

            # Show image
            cv2.imshow('RealSense RGB Image', color_image)
        
        # Process gyro frame
        if gyro_frame:
            gyro_data = gyro_frame.as_motion_frame().get_motion_data()
            print(f"Gyro: x={gyro_data.x:.5f}, y={gyro_data.y:.5f}, z={gyro_data.z:.5f}")
        
        # Process accel frame
        if accel_frame:
            accel_data = accel_frame.as_motion_frame().get_motion_data()
            print(f"Accel: x={accel_data.x:.5f}, y={accel_data.y:.5f}, z={accel_data.z:.5f}")

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    pipeline.stop()
    cv2.destroyAllWindows()