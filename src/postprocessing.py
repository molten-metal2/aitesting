import numpy as np
import cv2
from config import (
    CONF_THRESH,
    BANANA_CLASS,
    NMS_IOU_THRESH,
    DEBUG_MODE

)

def nms(boxes, scores, iou_threshold=0.5):
    """
    Performs Non-Maximum Suppression (NMS).
    Used to eliminate overlapping bounding boxes and keep only the best detection for each object.

    Pure NumPy Non-Maximum Suppression with division-by-zero protection.
    boxes: (N,4) xyxy
    scores: (N,)
    Returns indices of kept boxes.
    """
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []

    while order.size > 0:
        i = order[0]
        keep.append(i)

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        # Add epsilon to prevent division by zero
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def postprocess(output, frame):
    """
    Processes the raw output from the model.
    Used to decode predictions, filter by class (banana) and confidence, apply NMS, and draw bounding boxes on the frame.
    """
    h, w = frame.shape[:2]
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
        return frame, 0

    boxes_xyxy = boxes_xyxy[mask]
    scores     = scores[mask]
    cls_ids    = best_cls_ids[mask]

    # Keep only bananas
    banana_mask = cls_ids == BANANA_CLASS
    boxes_xyxy = boxes_xyxy[banana_mask]
    scores     = scores[banana_mask]

    if len(scores) == 0:
        return frame, 0
    
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

    # Draw kept boxes
    for i in keep:
        x1, y1, x2, y2 = boxes_xyxy[i]

        # Scale to frame
        x1 *= (w / 640)
        y1 *= (h / 640)
        x2 *= (w / 640)
        y2 *= (h / 640)

        x1 = int(x1); y1 = int(y1)
        x2 = int(x2); y2 = int(y2)

        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)
        cv2.putText(frame, "banana", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

    return frame, banana_count

