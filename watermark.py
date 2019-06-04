#!/usr/bin/env python3

"""
Pack all things related watermarking.

needs
    python3-opencv
"""

from datetime import datetime

# import numpy as np
import cv2 as cv


class Watermark():
    """docstring for Watermark."""

    WINDOWNAME = "Watermark Preview"

    def __init__(
            self,
            camera_device=0,
            filename_template="./image_{date_part}.png",
    ):
        """Initialize Instance."""
        super(Watermark, self).__init__()

        self.cap = None
        self.camera_device = camera_device

        self.save_next_frame_flag = False
        self.filename_full = "./test.png"
        self.filename_template = filename_template

    def __del__(self):
        """Clean up."""
        self.stop()

    def start(self):
        """Starts."""
        # cv.namedWindow(self.WINDOWNAME, cv.WINDOW_NORMAL)
        # cv.namedWindow(self.WINDOWNAME, cv.WINDOW_AUTOSIZE)
        cv.namedWindow(self.WINDOWNAME, cv.WINDOW_FULLSCREEN)
        # cv.setWindowProperty(
        #     self.WINDOWNAME,
        #     cv.WND_PROP_FULLSCREEN,
        #     cv.WINDOW_FULLSCREEN)
        self.cap = cv.VideoCapture(self.camera_device)
        # set configuration for camera
        # self.cap.set(cv.CAP_PROP_FPS, 30)
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap.set(cv.CV_CAP_PROP_CONVERT_RGB, True)
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
        while run:
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            frame_mod = frame
            # Our operations on the frame come here
            # frame_mod = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if self.save_next_frame_flag:
                self.save_next_frame_flag = False
                cv.imwrite(self.filename_full, frame)
                print("saved '" + self.filename_full + "'")

            # Display the resulting frame
            cv.imshow(self.WINDOWNAME, frame_mod)
            # the & 0xFF is needed for 64-bit machines
            key = cv.waitKey(1) & 0xFF
            run = not self.check_exit_key(key)
            if key == ord('s'):
                self.save_next_frame()

    def save_next_frame(self):
        """Save Next Captured Frame."""
        date_part = str(datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        # print(date_part)
        # print(self.filename_template)
        self.filename_full = self.filename_template.format(
            date_part=date_part
        )
        self.save_next_frame_flag = True

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

    input_filename_default = "./watermark.png"

    parser = argparse.ArgumentParser(
        description="test watermark."
    )

    parser.add_argument(
        "-w",
        "--watermark",
        help="specify watermark image. (defaults to {})".format(
            input_filename_default
        ),
        metavar='INPUT_FILENAME',
        default=input_filename_default
    )
    args = parser.parse_args()

    # print(args.output_filename)
    watermark = Watermark(
        camera_device=args.camera_device,
        filename_template=args.input_filename[0]
    )
    watermark.start()
    watermark.run()


if __name__ == "__main__":
    main()
