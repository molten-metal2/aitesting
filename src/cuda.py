import ctypes
from config import (
    cudart,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
)

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