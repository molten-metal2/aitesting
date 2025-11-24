import cv2
import ctypes
import numpy as np

from config import (
    INPUT_W,
    INPUT_H
)

def preprocess(frame):
    """
    Prepares an input image frame for inference.
    Used to resize, color convert, normalize, and transpose the image to match the model's input requirements.
    
    Resize → BGR→RGB → normalize → CHW → batch dimension
    """
    img = cv2.resize(frame, (INPUT_W, INPUT_H))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # CHW
    img = np.expand_dims(img, 0).copy()  # (1,3,640,640)
    return img