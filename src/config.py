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
