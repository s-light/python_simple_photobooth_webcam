#!/usr/bin/env python3

"""
Python & OpenCV based Photobooth.

needs
    python3-opencv
"""

import os
# import subprocess
from contextlib import contextmanager


##########################################
# functions

@contextmanager
def cd(newdir):
    """
    Change directory.

    found at:
    http://stackoverflow.com/questions/431684/how-do-i-cd-in-python/24176022#24176022
    """
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


##########################################
# classes

class PhotoBoothMain():
    """docstring for PhotoBoothMain."""

    def __init__(self):
        """Initialize Instance."""
        # print("PhotoBoothMain")
        self.path_script = os.path.dirname(os.path.abspath(__file__))
        print("path_script", self.path_script)

        super(PhotoBoothMain, self).__init__()

    def __del__(self):
        """Clean up."""
        pass

    def _run(self):
        """Run Photobooth."""
        pass

    def start(self):
        """Start Photobooth."""
        self._run()

    def stop(self):
        """Stop Photobooth."""
        pass

##########################################
# globals

# myPBHandler = None


##########################################
# main

def main():
    """Run Main SW."""
    import sys
    import argparse

    print(42*'*')
    print('Python Version: ' + sys.version)
    print(42*'*')

    input_filename_default = "./image1.png"
    output_filename_default = "./image1_wm.png"
    # ./captured/photo_booth-%Y%m%d-%H%M%S.%C"
    watermark_filename_default = "./mark.png"

    parser = argparse.ArgumentParser(
        description="adds watermark to image file."
    )

    parser.add_argument(
        "-i",
        "--input_filename",
        help="specify a location for the input file (defaults to {})".format(
            input_filename_default
        ),
        metavar='INPUT_FILENAME',
        default=input_filename_default
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
    parser.add_argument(
        "-w",
        "--watermark_filename",
        help=(
            "specify a location for the watermark file "
            "(defaults to {})"
        ).format(
            watermark_filename_default
        ),
        metavar='WATERMARK_FILENAME',
        default=watermark_filename_default
    )
    args = parser.parse_args()


if __name__ == "__main__":
    main()
