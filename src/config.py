# --------- CONFIG ---------
ENGINE_PATH   = "/home/david/yolov8n.engine"
INPUT_W       = 640
INPUT_H       = 640
CONF_THRESH   = 0.50  # Increased from 0.30 to reduce false positives
BANANA_CLASS = 46
NMS_IOU_THRESH = 0.45  # IoU threshold for NMS - lower = more aggressive suppression
DEBUG_MODE    = True   # Set to False to disable debug output

