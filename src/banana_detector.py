import cv2
import numpy as np
import ctypes
import ctypes.util

from config import (
    ENGINE_PATH,
    INPUT_W,
    INPUT_H,
    CONF_THRESH,
    BANANA_CLASS,
    NMS_IOU_THRESH,
    DEBUG_MODE,
    cudart,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
)

from cuda import (
    cuda_check, 
    cuda_malloc, 
    cuda_free, 
    cuda_memcpy
)

from tensorrt_engine import (
    load_engine,
    make_context_and_buffers
)


# ========== PRE/POST PROCESSING ==========

def preprocess(frame):
    """
    Prepares an input image frame for inference.
    Used to resize, color convert, normalize, and transpose the image to match the model's input requirements.
    
    Resize → BGR→RGB → normalize → CHW → batch dimension
    """
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # CHW
    img = np.expand_dims(img, 0).copy()  # (1,3,640,640)
    return img

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




# ========== MAIN LOOP ==========

def main():
    """
    Main entry point of the application.
    Used to initialize the engine, setup the camera, and run the real-time inference loop.
    """
    print("Loading TensorRT engine...")
    engine = load_engine(ENGINE_PATH)
    ctx, input_name, output_name, in_shape, out_shape, d_input, d_output = make_context_and_buffers(engine)

    print("Input tensor:", input_name, "shape:", in_shape)
    print("Output tensor:", output_name, "shape:", out_shape)

    # GStreamer pipeline for your Pi Cam v2 on Jetson
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
        print("Camera failed to open.")
        cuda_free(d_input)
        cuda_free(d_output)
        return

    print("Running inference... press 'q' in the window to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            img = preprocess(frame)  # (1,3,640,640) float32
            nbytes_in = img.nbytes

            # Host -> Device
            cuda_memcpy(
                d_input,
                img.ctypes.data_as(ctypes.c_void_p),
                nbytes_in,
                cudaMemcpyHostToDevice,
            )

            # Tell TensorRT where the GPU buffers are
            ctx.set_tensor_address(input_name, int(d_input.value))
            ctx.set_tensor_address(output_name, int(d_output.value))

            # Inference on CUDA stream 0
            ctx.execute_async_v3(stream_handle=0)
            cuda_check(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")

            # Device -> Host
            host_out = np.empty(out_shape, dtype=np.float32)
            cuda_memcpy(
                host_out.ctypes.data_as(ctypes.c_void_p),
                d_output,
                host_out.nbytes,
                cudaMemcpyDeviceToHost,
            )

            frame, banana_count = postprocess(host_out, frame)
            cv2.putText(frame, f"bananas: {banana_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            cv2.imshow("TRT Banana Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        cuda_free(d_input)
        cuda_free(d_output)
        print("Clean exit.")


if __name__ == "__main__":
    main()
