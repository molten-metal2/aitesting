"""UI rendering and display functions."""

import cv2


def render_frame(frame, banana_count):
    """
    Render detection results on frame.
    Returns the annotated frame.
    """
    cv2.putText(frame, f"bananas: {banana_count}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return frame


def display_frame(frame, window_name="TRT Banana Detector"):
    """
    Display frame and handle user input.
    Returns True if quit signal received, False otherwise.
    """
    cv2.imshow(window_name, frame)
    return cv2.waitKey(1) & 0xFF == ord('q')
