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
from ui import render_frame, display_frame
from errors import InferenceError


def run_inference_loop(cap, ctx, input_name, output_name, in_shape, out_shape, d_input, d_output):
    """
    Run the main inference loop.
    Captures frames, preprocesses, runs inference, and displays results.
    Raises InferenceError if inference fails.
    """
    print("Running inference... press 'q' in the window to quit.")
    
    try:
        while True:
            try:
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
                frame = render_frame(frame, banana_count)
                if display_frame(frame):
                    break
            except Exception as e:
                raise InferenceError(f"Inference loop error: {e}") from e
    finally:
        cap.release()
        cv2.destroyAllWindows()
