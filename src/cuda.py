import ctypes
import ctypes.util

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
    Used to ensure that CUDA operations completed successfully.
    """
    if status != 0:
        raise RuntimeError(f"{where} failed with error code {status}")


def cuda_malloc(nbytes):
    """
    Allocates memory and holds the address of the allocated memory on the GPU.
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
    Used to transfer input images to the GPU and retrieve inference results back to the CPU.
    """
    cuda_check(cudart.cudaMemcpy(dst, src, nbytes, kind), "cudaMemcpy")


def cuda_synchronize():
    """
    Synchronize with CUDA device to ensure all operations complete.
    Includes error checking.
    """
    cuda_check(cudart.cudaDeviceSynchronize(), "cudaDeviceSynchronize")