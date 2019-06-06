#!/bin/bash

DISPLAY=:0.0 \
XAUTHORITY=/home/pi/.Xauthority \
/usr/bin/python3
/home/pi/python_simple_photobooth_webcam/cam.py \
--camera-device="/dev/video_cam_C1" \
--output_filename="/home/pi/python_simple_photobooth_webcam/captured/test_{date_part}.jpg"
