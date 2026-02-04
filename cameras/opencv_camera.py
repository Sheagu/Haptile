from typing import Optional, Tuple

import cv2
import numpy as np


class OpenCVCamera:
    """Simple OpenCV camera driver for webcams."""

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """Initialize the OpenCV camera.

        Args:
            camera_id: The camera device ID (usually 0, 1, 2, etc.)
            width: The width of the camera frame.
            height: The height of the camera frame.
        """
        self.camera_id = camera_id
        self.cap = cv2.VideoCapture(camera_id)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera {camera_id}")
        
        # Set camera resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Camera {camera_id} initialized: {self.width}x{self.height}")

    def read(
        self, img_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera.

        Args:
            img_size: The size of the image to return. If None, the original size is returned.

        Returns:
            np.ndarray: The color image in RGB format (H, W, 3).
            np.ndarray: Empty depth image (H, W) - webcams don't have depth.
        """
        ret, frame = self.cap.read()
        
        if not ret:
            raise RuntimeError(f"Failed to read from camera {self.camera_id}")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if img_size is not None:
            frame_rgb = cv2.resize(frame_rgb, img_size)
        
        # Create empty depth image (webcams don't have depth)
        depth = np.zeros((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.uint16)
        
        return frame_rgb, depth

    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            print(f"Camera {self.camera_id} released")

    def __del__(self):
        """Destructor to ensure camera is released."""
        self.release()
