"""Camera initialization and management."""

import cv2
from errors import CameraInitializationError


def initialize_camera():
    """
    Initialize camera capture with GStreamer pipeline.
    Returns the VideoCapture object.
    Raises CameraInitializationError if initialization fails.
    """
    gst = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=2 ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise CameraInitializationError("Camera failed to open.")
    return cap
