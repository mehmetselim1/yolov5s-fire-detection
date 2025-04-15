import cv2
import time
import numpy as np


class Camera:
    def __init__(self, camera_id=0, width=640, height=480):
        """
        Initialize camera interface

        Args:
            camera_id: Camera device ID (0 for default)
            width: Frame width
            height: Frame height
        """
    
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.cap = None
        
    def open(self):
        """
        Open camera connection
        """
        # Initializa camera with gstreamer for Jetson compatibility
        gst_str = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/xraw, format=BGRx ! "
            f"videoconver ! video/x-raw, format=BGR ! appsink"
        )
        
        try:
            
            # Try using gstreamer pipeline first (for Raspberry Pi v2 on Jetson)
            self.cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            if not self.cap.isOpened():
                # Fallback to standard capture
                self.cap = cv2.VideoCapture(self.camera_id)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            if not self.cap.isOpened():
                raise Exception("Failed to open camera")
            
            return True
        
        except Exception as e:
            print(f"Error opening camera: {e}")
            return False
        
        
    def read(self):
        """Read a frame from camera"""
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                return True, frame
        return False, None
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
