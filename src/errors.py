"""Custom exception classes for the TensorRT Banana Detector."""


class BananaDetectorError(Exception):
    """Base exception for all detector-related errors."""
    pass


class EngineInitializationError(BananaDetectorError):
    """Raised when TensorRT engine fails to initialize."""
    pass


class CameraInitializationError(BananaDetectorError):
    """Raised when camera initialization fails."""
    pass


class InferenceError(BananaDetectorError):
    """Raised when inference fails."""
    pass


class GPUMemoryError(BananaDetectorError):
    """Raised when GPU memory operations fail."""
    pass


class FrameCaptureError(BananaDetectorError):
    """Raised when frame capture fails."""
    pass
