#!/bin/bash

base="/home/pi/python_simple_photobooth_webcam"

DISPLAY=:0.0 \
XAUTHORITY=/home/pi/.Xauthority \
/usr/bin/python3 \
    $base/cam.py \
    --camera-device="/dev/video_cam_C1" \
    --output_filename="$base/captured/img_{date_part}.png" \
    --overlay_filename="$base/overlay/picture_frame.png" \
    --fullscreen
