#!/usr/bin/env python3

"""
simple webcam based python / opencv photobooth.

original this file was meant to pack all things related to camera...
but for now it is what it is ;-)

needs
    python3-opencv
    numpy
"""

from datetime import datetime

import numpy as np
import cv2 as cv


class Cam():
    """docstring for Cam."""

    WINDOWNAME = "Photobooth"

    def __init__(
            self,
            camera_device=0,
            output_filename_template="./image_{date_part}.png",
            overlay_filename="./overlay/picture_frame__HelloWorld.png",
            fullscreen=0,
    ):
        """Initialize Instance."""
        super(Cam, self).__init__()

        self.cap = None
        self.camera_device = camera_device

        self.save_next_frame_flag = False
        self.show_last_saved_frame_flag = False
        self.output_filename_full = "./test.png"
        # self.output_filename_full_debug = "./test_debug.png"
        self.output_filename_template = output_filename_template
        self.overlay_filename = overlay_filename
        self.overlay_position = (-107, -142)

        self.frame_size = (1920, 1080)
        self.result_image_size = self.frame_size
        self.result_image = False
        self.frame = False
        # self.preview_window_size = (1280, 1024)
        # self.preview_size = self.preview_window_size

        self.fullscreen = fullscreen

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
        if self.fullscreen:
            cv.setWindowProperty(
                self.WINDOWNAME,
                cv.WND_PROP_FULLSCREEN,
                cv.WINDOW_FULLSCREEN)
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
        # print("CAP_PROP_FOURCC", self.cap.get(cv.CAP_PROP_FOURCC))
        self.cap.set(cv.CAP_PROP_FPS, 30)
        # self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        # self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, self.frame_size[0])
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, self.frame_size[1])
        # self.cap.set(cv.CAP_PROP_CONVERT_RGB, True)

        self.save_next_frame_flag = False
        self.show_last_saved_frame_flag = False

        self.load_overlay_and_prepare_result_image()

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
        ret, self.frame = self.cap.read()
        height, width = self.frame.shape[:2]
        self.preview_size = (width//2, height//2)
        frame_preview = cv.resize(self.frame, self.preview_size)
        while run:
            # frame_old = frame
            # Capture frame-by-frame
            ret, self.frame = self.cap.read()
            frame_preview = cv.resize(self.frame, self.preview_size)

            if self.save_next_frame_flag:
                self.update_result_image()
                cv.imwrite(self.output_filename_full, self.result_image)
                print("saved as '" + self.output_filename_full + "'")
                self.show_last_saved_frame_flag = True
                self.save_next_frame_flag = False

            # update preview window
            if self.show_last_saved_frame_flag:
                cv.imshow(self.WINDOWNAME, self.result_image)
                # the & 0xFF is needed for 64-bit machines
                key = cv.waitKey(5000) & 0xFF
                self.show_last_saved_frame_flag = False
            else:
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
        # print("output_filename_template", self.output_filename_template)
        self.output_filename_full = self.output_filename_template.format(
            date_part=date_part
        )
        # self.output_filename_full_debug = self.output_filename_template.
        # format(
        #     date_part=date_part + '_debug'
        # )
        # print("output_filename_full", self.output_filename_full)
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

    def load_overlay_and_prepare_result_image(self):
        """Load overlay image."""
        self.overlay_img = cv.imread(
            self.overlay_filename, cv.IMREAD_UNCHANGED)
        # calculate result image size
        overlay_height, overlay_width = self.overlay_img.shape[:2]
        # result_height, result_width = self.frame_size
        # # handle bigger overlays then frame
        # x, y = self.overlay_pos
        # if x < 0:
        #     result_width += x
        # if y < 0:
        #     result_height += y
        # #
        # if result_width < overlay_width:
        #     result_width = overlay_width
        # if result_height < overlay_height:
        #     result_height = overlay_height
        # self.result_image_size = (result_width, result_height)
        # TODO(s-light): fix this calculation..
        # for now we just assume the overlay is bigger ;-)
        self.result_image_size = (overlay_height, overlay_width)
        # so we also need to move the frame and not the overlay
        self.frame_position = (
            self.overlay_position[0] * -1, self.overlay_position[1] * -1)
        self.overlay_position = 0, 0
        self.reset_result_image()

    def reset_result_image(self):
        """Reset result_image to transparent."""
        # create transparent image
        # thanks for the example from
        # https://stackoverflow.com/a/44595221/574981
        # RGBA == 4
        n_channels = 4
        height, width = self.result_image_size
        self.result_image = np.zeros(
            (height, width, n_channels), dtype=np.uint8)

    def update_result_image(self):
        """Update result image."""
        self.reset_result_image()
        frame = cv.cvtColor(self.frame, cv.COLOR_RGB2RGBA).copy()
        self.overlay_image_alpha(
            self.result_image, frame, self.frame_position)
        self.overlay_image_alpha(
            self.result_image, self.overlay_img, self.overlay_position)

    def overlay_image_alpha(img, img_overlay, pos):
        """
        Overlay img_overlay on top of img at position.

        this is based on the answer from Mateen Ulhaq
        https://stackoverflow.com/a/45118011/574981
        """
        x, y = pos

        # create alpha_mask
        alpha_mask = img_overlay[:, :, 3] / 255.0

        # Image ranges
        y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
        x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

        # Overlay ranges
        y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
        x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

        # Exit if nothing to do
        if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
            return

        channels = img.shape[2]

        alpha = alpha_mask[y1o:y2o, x1o:x2o]
        alpha_inv = 1.0 - alpha

        for c in range(channels):
            img[y1:y2, x1:x2, c] = (
                alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                alpha_inv * img[y1:y2, x1:x2, c]
            )

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
    print(__doc__)
    print(42*'*')

    # print(42*'*')
    # print('debug info')
    # events = [i for i in dir(cv) if 'EVENT' in i]
    # print(events)
    # print(42*'*')

    cam = '/dev/video_cam_C1'
    output_filename_default = "./captured/test_{date_part}.png"
    overlay_filename_default = "./mask/picture_frame__HelloWorld.png"

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
    parser.add_argument(
        "-w",
        "--overlay_filename",
        help=(
            "specify a location for the overlay file "
            "(defaults to {})"
        ).format(
            overlay_filename_default
        ),
        metavar='OVERLAY_FILENAME',
        default=overlay_filename_default
    )
    parser.add_argument(
        "-f",
        "--fullscreen",
        help="start in fullscreen mode",
        action="store_true"
    )
    args = parser.parse_args()

    print(args.output_filename)
    cam = Cam(
        camera_device=args.camera_device,
        output_filename_template=args.output_filename,
        overlay_filename=args.overlay_filename,
        fullscreen=args.fullscreen
    )
    cam.start()
    cam.run()


if __name__ == "__main__":
    main()
