#!/bin/bash
export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvarguscamerasrc.so:/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgstnvvidconv.so
python3 src/banana_detector.py "$@"
