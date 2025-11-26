"""Core inference loop orchestration."""

import cv2
import numpy as np
import ctypes

from cuda import (
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cuda_memcpy,
    cuda_synchronize
)

from preprocessing import preprocess
from postprocessing import postprocess
from ui import render_frame, display_frame, draw_detections
from errors import InferenceError


def run_inference_loop(cap, trt_state):
    """
    Run the main inference loop.
    Captures frames, preprocesses, runs inference, and displays results.
    Raises InferenceError if inference fails.
    """
    print("Running inference... press 'q' in the window to quit.")
    
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            img = preprocess(frame)  # (1,3,640,640) float32
            nbytes_in = img.nbytes

            # Host -> Device
            cuda_memcpy(
                trt_state.d_input,
                img.ctypes.data_as(ctypes.c_void_p),
                nbytes_in,
                cudaMemcpyHostToDevice,
            )

            # Tell TensorRT where the GPU buffers are
            trt_state.ctx.set_tensor_address(trt_state.input_name, int(trt_state.d_input.value))
            trt_state.ctx.set_tensor_address(trt_state.output_name, int(trt_state.d_output.value))

            # Inference on CUDA stream 0
            trt_state.ctx.execute_async_v3(stream_handle=0)
            cuda_synchronize()

            # Device -> Host
            host_out = np.empty(trt_state.out_shape, dtype=np.float32)
            cuda_memcpy(
                host_out.ctypes.data_as(ctypes.c_void_p),
                trt_state.d_output,
                host_out.nbytes,
                cudaMemcpyDeviceToHost,
            )

            # Get frame dimensions for postprocessing
            frame_shape = frame.shape[:2]  # (height, width)
            boxes, banana_count = postprocess(host_out, frame_shape)
            
            # Apply visualization
            frame = draw_detections(frame, boxes, banana_count)
            frame = render_frame(frame, banana_count)
            if display_frame(frame):
                break
        except Exception as e:
            raise InferenceError(f"Inference loop error: {e}") from e
