"""UI rendering and display functions."""

import cv2


def draw_detections(frame, boxes, banana_count, box_color=(0, 255, 255), box_thickness=2, label_color=(0, 255, 255)):
    """
    Draw bounding boxes and labels on frame.
    Separates visualization from data processing logic (SRP).
    
    Args:
        frame: Input frame to draw on
        boxes: List of bounding boxes in (x1, y1, x2, y2) format
        banana_count: Number of detected bananas
        box_color: BGR tuple for box color (default: yellow)
        box_thickness: Thickness of box outline (default: 2)
        label_color: BGR tuple for label text color (default: yellow)
    
    Returns:
        Annotated frame with drawn boxes and labels.
    """
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)
        cv2.putText(frame, "banana", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, label_color, 2)
    
    return frame


def render_frame(frame, banana_count):
    """
    Render detection count on frame.
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
