#!/usr/bin/env python3

"""
Pack all things related to camera.

needs
    python3-opencv
"""

# import numpy as np
import cv2 as cv


class Cam():
    """docstring for Cam."""

    def __init__(self, camera_device=0):
        """Initialize Instance."""
        super(Cam, self).__init__()

        self.cap = None
        self.camera_device = camera_device

    def __del__(self):
        """Clean up."""
        self.stop()

    def start(self):
        """Stop and free resources."""
        self.cap = cv.VideoCapture(self.camera_device)
        # set configuration for camera
        # self.cap.set(cv.CAP_PROP_FPS, 30)
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        # self.cap.set(cv.CV_CAP_PROP_CONVERT_RGB, True)

    def stop(self):
        """Stop and free resources."""
        # When everything done, release the capture
        if self.cap:
            self.cap.release()
            cv.destroyAllWindows()

    def check_exit_key(self):
        """Check for control Keys."""
        result = False
        key = cv.waitKey(1)
        # 'q' == quit
        if key & 0xFF == ord('q'):
            result = True
        # ESC
        elif key == 27:
            result = True
        return result

    def run(self):
        """Run Photobooth."""
        while(True):
            # Capture frame-by-frame
            ret, frame = self.cap.read()
            frame_mod = frame
            # Our operations on the frame come here
            # frame_mod = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            # Display the resulting frame
            cv.imshow('frame', frame_mod)
            if self.check_key():
                break

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

    cam = 0

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
    args = parser.parse_args()

    cam = Cam(args.camera_device)
    cam.start()
    cam.run()


if __name__ == "__main__":
    main()
