import numpy as np
import tensorrt as trt
from cuda import cuda_malloc, cuda_free
from errors import EngineInitializationError


class TensorRTState:
    """
    Encapsulates the state required for TensorRT inference.
    """
    def __init__(self, ctx, input_name, output_name, in_shape, out_shape, d_input, d_output):
        self.ctx = ctx
        self.input_name = input_name
        self.output_name = output_name
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.d_input = d_input
        self.d_output = d_output

    def cleanup(self):
        """Frees the allocated GPU memory."""
        if self.d_input:
            cuda_free(self.d_input)
        if self.d_output:
            cuda_free(self.d_output)


# ========== TENSORRT SETUP ==========

def initialize_engine(engine_path):
    """
    Initialize TensorRT engine and GPU buffers.
    Returns a TensorRTState object containing context, tensor names, shapes, and device pointers.
    Raises EngineInitializationError if initialization fails.
    """
    try:
        print("Loading TensorRT engine...")
        engine = load_engine(engine_path)
        trt_state = make_context_and_buffers(engine)
        
        print("Input tensor:", trt_state.input_name, "shape:", trt_state.in_shape)
        print("Output tensor:", trt_state.output_name, "shape:", trt_state.out_shape)
        
        return trt_state
    except Exception as e:
        raise EngineInitializationError(f"Failed to initialize TensorRT engine: {e}") from e

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

    return TensorRTState(ctx, input_name, output_name, in_shape, out_shape, d_input, d_output)
