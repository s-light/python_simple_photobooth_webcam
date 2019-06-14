#!/bin/bash

base="/home/pi/python_simple_photobooth_webcam"
# capture="/home/pi/python_simple_photobooth_webcam/captured/img_{date_part}.png"
# capture="/media/pi/STEFAN64/captured/img_{date_part}.png"
capture="./captured/img_{date_part}.png"
overlay="./overlay/picture_frame__useme.png"

DISPLAY=:0.0 \
XAUTHORITY=/home/pi/.Xauthority \
/usr/bin/python3 \
    $base/cam.py \
    --device="/dev/video_cam_C1" \
    --output_filename="$capture" \
    --overlay_filename="$overlay" \
    --fullscreen
