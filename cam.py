#!/usr/bin/env python3

"""
Pack all things related to camera.

needs
    python3-opencv
"""

from datetime import datetime

# import numpy as np
import cv2 as cv


class Cam():
    """docstring for Cam."""

    WINDOWNAME = "Photobooth"

    def __init__(
            self,
            camera_device=0,
            filename_template="./image_{date_part}.png",
    ):
        """Initialize Instance."""
        super(Cam, self).__init__()

        self.cap = None
        self.camera_device = camera_device

        self.save_next_frame_flag = False
        self.filename_full = "./test.png"
        self.filename_full_debug = "./test_debug.png"
        self.filename_template = filename_template
        # self.frame_size = (1920, 1080)
        # self.preview_window_size = (1280, 1024)
        # self.preview_size = self.preview_window_size

    def __del__(self):
        """Clean up."""
        self.stop()

    def mouse_handler(self, event, x, y, flags, param):
        """Handle Mouse Events."""
        if event == cv.EVENT_LBUTTONDBLCLK:
            self.toggle_fullscreen()

    def start(self):
        """Stop and free resources."""
        # setup window
        cv.namedWindow(self.WINDOWNAME, cv.WINDOW_NORMAL)
        # cv.namedWindow(self.WINDOWNAME, cv.WINDOW_AUTOSIZE)
        # cv.namedWindow(self.WINDOWNAME, cv.WINDOW_FULLSCREEN)
        # cv.setWindowProperty(
        #     self.WINDOWNAME,
        #     cv.WND_PROP_FULLSCREEN,
        #     cv.WINDOW_FULLSCREEN)
        cv.setWindowProperty(
            self.WINDOWNAME,
            cv.WND_PROP_OPENGL,
            cv.WINDOW_OPENGL)
        cv.setMouseCallback(self.WINDOWNAME, self.mouse_handler)

        # camera capture
        self.cap = cv.VideoCapture(self.camera_device)
        # set configuration for camera
        # self.cap.set(cv.CAP_PROP_FOURCC('M', 'J', 'P', 'G'))
        # self.cap.set(cv.CAP_PROP_FOURCC('MJPG'))
        # self.cap.set(cv.CAP_PROP_FOURCC, cv.CAP_PROP_FOURCC('MJPG'))
        print("CAP_PROP_FOURCC", self.cap.get(cv.CAP_PROP_FOURCC))
        self.cap.set(cv.CAP_PROP_FPS, 30)
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv.CAP_PROP_CONVERT_RGB, True)
        self.save_next_frame_flag = False

    def stop(self):
        """Stop and free resources."""
        # When everything done, release the capture
        if self.cap:
            self.cap.release()
            cv.destroyWindow(self.WINDOWNAME)

    def check_exit_key(self, key):
        """Check for control Keys."""
        result = False
        # 'q' == quit
        if key == ord('q'):
            result = True
        # ESC
        elif key == 27:
            result = True
        return result

    def run(self):
        """Run Photobooth."""
        run = True
        # Capture first frame.
        ret, frame = self.cap.read()
        height, width = frame.shape[:2]
        self.preview_size = (width//2, height//2)
        frame_mod = cv.resize(frame, self.preview_size)
        while run:
            frame_old = frame
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            # frame_mod = frame
            # Our operations on the frame come here
            # frame_mod = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            frame_preview = cv.resize(frame, self.preview_size)

            if self.save_next_frame_flag:
                self.save_next_frame_flag = False
                # print("save as '" + self.filename_full + "'")
                # cv.imwrite(self.filename_full, frame)
                frame_blend = cv.addWeighted(frame, 0.5, frame_old, 0.5, 0)
                cv.imwrite(self.filename_full, frame_blend)
                cv.imwrite(self.filename_full_debug + '', frame)
                # print("saved.")
                print("saved as '" + self.filename_full + "'")
                # print("shape", frame.shape)

            # Display the resulting frame
            cv.imshow(self.WINDOWNAME, frame_preview)
            # the & 0xFF is needed for 64-bit machines
            key = cv.waitKey(1) & 0xFF
            run = not self.check_exit_key(key)
            if key == ord('s'):
                self.save_next_frame()
            elif key == ord('f'):
                self.toggle_fullscreen()

    def save_next_frame(self):
        """Save Next Captured Frame."""
        date_part = str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        # print("date_part", date_part)
        # print("filename_template", self.filename_template)
        self.filename_full = self.filename_template.format(
            date_part=date_part
        )
        self.filename_full_debug = self.filename_template.format(
            date_part=date_part + '_debug'
        )
        # print("filename_full", self.filename_full)
        self.save_next_frame_flag = True

    def toggle_fullscreen(self):
        """Toggle Window Fullscreen."""
        state = cv.getWindowProperty(self.WINDOWNAME, cv.WND_PROP_FULLSCREEN)
        print('state', state)
        print('cv.WINDOW_NORMAL', cv.WINDOW_NORMAL)
        print('cv.WINDOW_FULLSCREEN', cv.WINDOW_FULLSCREEN)
        if (state == cv.WINDOW_NORMAL):
            cv.setWindowProperty(
                self.WINDOWNAME,
                cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_FULLSCREEN)
        else:
            cv.setWindowProperty(
                self.WINDOWNAME,
                cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_NORMAL)

##########################################


##########################################
# main

def main():
    """Run Main SW."""
    import sys
    import argparse

    print(42*'*')
    print('Python Version: ' + sys.version)
    print(42*'*')


    print(42*'*')
    print('debug info')
    events = [i for i in dir(cv) if 'EVENT' in i]
    print( events )
    print(42*'*')



    cam = '/dev/video_cam_C1'
    output_filename_default = "./captured/test_{date_part}.png"

    parser = argparse.ArgumentParser(
        description="test cam."
    )

    parser.add_argument(
        "-d",
        "--camera-device",
        help="specify camera device number to use. (defaults to {})".format(
            cam
        ),
        metavar='INPUT_FILENAME',
        default=cam
    )
    parser.add_argument(
        "-o",
        "--output_filename",
        help="specify a location for the output file (defaults to {})".format(
            output_filename_default
        ),
        metavar='OUTPUT_FILENAME',
        default=output_filename_default
    )
    args = parser.parse_args()

    print(args.output_filename)
    cam = Cam(
        camera_device=args.camera_device,
        filename_template=args.output_filename
    )
    cam.start()
    cam.run()


if __name__ == "__main__":
    main()
