import numpy as np
import tensorrt as trt
from cuda import cuda_malloc


# ========== TENSORRT SETUP ==========

def load_engine(path):
    """
    Loads and deserializes a TensorRT engine from a file.
    Used to load the pre-trained YOLO model.
    """
    logger = trt.Logger(trt.Logger.INFO)
    with open(path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize engine")
    return engine


def make_context_and_buffers(engine):
    """
    Used to setup the necessary environment and memory on the GPU for running the inference engine, 
    by creating an execution context and allocating input/output buffers.
    
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
