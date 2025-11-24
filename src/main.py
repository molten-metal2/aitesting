import cv2
import numpy as np
import ctypes

from config import (
    ENGINE_PATH,
    CONF_THRESH,
    BANANA_CLASS,
    NMS_IOU_THRESH,
    DEBUG_MODE,
)

from cuda import (
    cudart,
    cuda_malloc, 
    cuda_free, 
    cuda_memcpy,
    cuda_synchronize
)

from tensorrt_engine import (
    load_engine,
    make_context_and_buffers
)

from preprocessing import preprocess
from postprocessing import postprocess, nms

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
            cuda_synchronize()

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
