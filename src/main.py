import cv2

from config import ENGINE_PATH

from tensorrt_engine import initialize_engine

from errors import (
    EngineInitializationError,
    CameraInitializationError,
    InferenceError,
)

from camera import initialize_camera
from inference import run_inference_loop

# ========== MAIN LOOP ==========

def main():
    """
    Main entry point of the application.
    Used to initialize the engine, setup the camera, and run the real-time inference loop.
    """
    trt_state = None
    cap = None
    
    try:
        # Initialize engine
        try:
            trt_state = initialize_engine(ENGINE_PATH)
        except EngineInitializationError as e:
            print(f"Error: {e}")
            return
        
        # Initialize camera
        try:
            cap = initialize_camera()
        except CameraInitializationError as e:
            print(f"Error: {e}")
            return
        
        # Run inference loop
        try:
            run_inference_loop(cap, trt_state)
        except InferenceError as e:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup resources
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if trt_state:
            trt_state.cleanup()
        print("Clean exit.")


if __name__ == "__main__":
    main()
