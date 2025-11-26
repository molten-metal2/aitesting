import numpy as np
import cv2
from config import (
    CONF_THRESH,
    BANANA_CLASS,
    NMS_IOU_THRESH,
    DEBUG_MODE,
    INPUT_W,
    INPUT_H
)

def nms(boxes, scores, iou_threshold=0.5):
    """
    Performs Non-Maximum Suppression (NMS) using OpenCV's optimized C++ implementation.
    Used to eliminate overlapping bounding boxes and keep only the best detection for each object.

    Args:
        boxes: (N,4) array in xyxy format
        scores: (N,) array of confidence scores
        iou_threshold: IoU threshold for NMS
    
    Returns:
        List of indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    # Convert boxes from xyxy to xywh format for cv2.dnn.NMSBoxes
    # cv2.dnn.NMSBoxes expects float32 arrays: [x, y, width, height]
    boxes_xywh = np.zeros((len(boxes), 4), dtype=np.float32)
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        boxes_xywh[i] = [x1, y1, x2 - x1, y2 - y1]
    
    # Ensure scores are float32
    scores_float = np.array(scores, dtype=np.float32)
    
    # Apply OpenCV's optimized NMS
    # Returns array of indices of kept boxes
    keep_indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh.tolist(),
        scores=scores_float.tolist(),
        score_threshold=0.0,  # We already filtered by confidence
        nms_threshold=iou_threshold
    )
    
    # cv2.dnn.NMSBoxes returns a numpy array, convert to list for compatibility
    return keep_indices.flatten().tolist() if len(keep_indices) > 0 else []


def postprocess(output, frame_shape):
    """
    Processes the raw output from the model.
    Returns detected banana boxes and count (data processing only, no visualization).
    
    Args:
        output: Raw model output from TensorRT
        frame_shape: Tuple of (height, width) of the original frame
    
    Returns:
        Tuple of (boxes_xyxy, banana_count) where:
            - boxes_xyxy: List of bounding boxes in (x1, y1, x2, y2) format, scaled to frame dimensions
            - banana_count: Number of detected bananas after NMS
    """
    h, w = frame_shape
    out = output[0]

    # Fix shape: (84,8400) -> (8400,84)
    if out.shape == (84, 8400):
        pred = out.T
    else:
        pred = out

    # YOLOv8 outputs boxes in xywh format (center_x, center_y, width, height)
    # Convert to xyxy format (x1, y1, x2, y2)
    boxes_xywh = pred[:, :4]
    cx, cy, w_box, h_box = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = cx - w_box / 2
    y1 = cy - h_box / 2
    x2 = cx + w_box / 2
    y2 = cy + h_box / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    
    cls_scores = pred[:, 4:]

    best_cls_scores = cls_scores.max(axis=1)
    best_cls_ids    = cls_scores.argmax(axis=1)

    scores = best_cls_scores
    
    mask = scores > CONF_THRESH

    if not np.any(mask):
        return [], 0

    boxes_xyxy = boxes_xyxy[mask]
    scores     = scores[mask]
    cls_ids    = best_cls_ids[mask]

    # Keep only bananas
    banana_mask = cls_ids == BANANA_CLASS
    boxes_xyxy = boxes_xyxy[banana_mask]
    scores     = scores[banana_mask]

    if len(scores) == 0:
        return [], 0
    
    # Only print debug when we have banana detections
    if DEBUG_MODE and len(scores) > 0:
        print(f"\n[DEBUG] Banana detections before NMS: {len(scores)}")
        for i in range(min(5, len(scores))):  # Show max 5 boxes
            print(f"  Box {i}: conf={scores[i]:.3f}, bbox=[{boxes_xyxy[i][0]:.1f}, {boxes_xyxy[i][1]:.1f}, {boxes_xyxy[i][2]:.1f}, {boxes_xyxy[i][3]:.1f}]")
        if len(scores) > 5:
            print(f"  ... and {len(scores) - 5} more")

    # ---- Apply NMS here ----
    keep = nms(boxes_xyxy, scores, iou_threshold=NMS_IOU_THRESH)

    banana_count = len(keep)
    
    if DEBUG_MODE and len(scores) > 0:
        print(f"[DEBUG] After NMS (IoU={NMS_IOU_THRESH}): {banana_count} banana(s) kept")
        if banana_count != len(scores):
            print(f"[DEBUG] Suppressed {len(scores) - banana_count} duplicate detection(s)")
        print("-" * 60)

    # Scale boxes to frame dimensions and return
    scaled_boxes = []
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i]

        # Scale from model space to frame dimensions
        x1 *= (w / INPUT_W)
        y1 *= (h / INPUT_H)
        x2 *= (w / INPUT_W)
        y2 *= (h / INPUT_H)

        scaled_boxes.append([int(x1), int(y1), int(x2), int(y2)])

    return scaled_boxes, banana_count

