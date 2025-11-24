from config import ENGINE_PATH

from cuda import cuda_free

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
    ctx = None
    d_input = None
    d_output = None
    cap = None
    
    try:
        # Initialize engine
        try:
            ctx, input_name, output_name, in_shape, out_shape, d_input, d_output = initialize_engine(ENGINE_PATH)
        except EngineInitializationError as e:
            print(f"Error: {e}")
            return
        
        # Initialize camera
        try:
            cap = initialize_camera()
        except CameraInitializationError as e:
            print(f"Error: {e}")
            cuda_free(d_input)
            cuda_free(d_output)
            return
        
        # Run inference loop
        try:
            run_inference_loop(cap, ctx, input_name, output_name, in_shape, out_shape, d_input, d_output)
        except InferenceError as e:
            print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Cleanup resources
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        if d_input:
            cuda_free(d_input)
        if d_output:
            cuda_free(d_output)
        print("Clean exit.")


if __name__ == "__main__":
    main()
