import cv2
import numpy as np
import tensorrt as trt
import ctypes
import ctypes.util

# --------- CONFIG ---------
ENGINE_PATH   = "/home/david/yolov8n.engine"
INPUT_W       = 640
INPUT_H       = 640
CONF_THRESH   = 0.50  # Increased from 0.30 to reduce false positives
BANANA_CLASS = 46
NMS_IOU_THRESH = 0.45  # IoU threshold for NMS - lower = more aggressive suppression
DEBUG_MODE    = True   # Set to False to disable debug output
# --------------------------


# ========== CUDA RUNTIME VIA CTYPES (NO PYCUDA) ==========
# Find and load libcudart.so
libcudart_path = ctypes.util.find_library("cudart")
if libcudart_path is None:
    raise RuntimeError("Could not find libcudart.so (CUDA runtime). Is CUDA installed and in ldconfig?")

cudart = ctypes.CDLL(libcudart_path)

# cudaMemcpy kinds
cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2

# Set function signatures
cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype  = ctypes.c_int

cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaFree.restype  = ctypes.c_int

cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                              ctypes.c_size_t, ctypes.c_int]
cudart.cudaMemcpy.restype  = ctypes.c_int

cudart.cudaDeviceSynchronize.argtypes = []
cudart.cudaDeviceSynchronize.restype  = ctypes.c_int


def cuda_check(status, where="CUDA call"):
    """
    Checks the return status of a CUDA API call.
    Used to ensure that CUDA operations (like memory allocation or copying) completed successfully.
    Raises a RuntimeError if the status indicates an error.
    """
    if status != 0:
        raise RuntimeError(f"{where} failed with error code {status}")


def cuda_malloc(nbytes):
    """
    Allocates memory on the GPU.
    Used to create buffers for input and output tensors on the device before inference.
    """
    ptr = ctypes.c_void_p()
    cuda_check(cudart.cudaMalloc(ctypes.byref(ptr), nbytes), "cudaMalloc")
    return ptr


def cuda_free(ptr):
    """
    Frees allocated GPU memory.
    Used to clean up resources and prevent memory leaks after the program finishes.
    """
    if ptr:
        cuda_check(cudart.cudaFree(ptr), "cudaFree")


def cuda_memcpy(dst, src, nbytes, kind):
    """
    Copies memory between Host (CPU) and Device (GPU).
    Used to transfer input images to the GPU and retrieve inference results back to the CPU.
    """
    cuda_check(cudart.cudaMemcpy(dst, src, nbytes, kind), "cudaMemcpy")


# ========== TENSORRT SETUP ==========

def load_engine(path):
    """
    Loads and deserializes a TensorRT engine from a file.
    Used to load the pre-trained YOLO model optimized for inference on the GPU.
    """
    logger = trt.Logger(trt.Logger.INFO)
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize engine")
    return engine


def make_context_and_buffers(engine):
    """
    Creates an execution context and allocates input/output buffers.
    Used to setup the necessary environment and memory on the GPU for running the inference engine.
    
    Use new TensorRT API:
      - get_tensor_name
      - get_tensor_shape
      - get_tensor_dtype
      - set_tensor_address
      - execute_async_v3
    """
    ctx = engine.create_execution_context()

    # We know there are 2 tensors: input and output
    input_name  = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)

    in_shape  = engine.get_tensor_shape(input_name)   # (1, 3, 640, 640)
    out_shape = engine.get_tensor_shape(output_name)  # (1, 84, 8400)

    # Allocate GPU buffers
    in_size  = int(np.prod(in_shape)) * np.dtype(np.float32).itemsize
    out_size = int(np.prod(out_shape)) * np.dtype(np.float32).itemsize

    d_input  = cuda_malloc(in_size)
    d_output = cuda_malloc(out_size)

    return ctx, input_name, output_name, in_shape, out_shape, d_input, d_output


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
