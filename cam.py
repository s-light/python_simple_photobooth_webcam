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

        self.frame_size = (1080, 1920)
        self.result_image_size = self.frame_size
        self.result_image = None
        self.frame = None
        self.frame_preview = None
        self.show_preview = True
        # self.preview_window_size = (1280, 1024)
        # self.frame_preview_size = self.preview_window_size

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

        frame_height, frame_width = self.frame_size
        # print('self.frame_size', self.frame_size)
        # print('self.frame_size[0] → height', frame_height)
        # print('self.frame_size[1] → width', frame_width)

        self.save_next_frame_flag = False
        self.show_last_saved_frame_flag = False

        self.frame_preview_size = (frame_height//2, frame_width//2)
        # print('self.frame_preview_size', self.frame_preview_size)
        self.frame_preview_size_4resize = (
            self.frame_preview_size[1], self.frame_preview_size[0])
        # print('frame_preview_size_4resize', self.frame_preview_size_4resize)

        prev_height, prev_width = self.frame_preview_size
        if not self.fullscreen:
            cv.resizeWindow(self.WINDOWNAME, prev_width, prev_height)

        print('-')
        self.init_frame_buffers()

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
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, frame_height)
        # self.cap.set(cv.CAP_PROP_CONVERT_RGB, True)
        # precapture some frames..
        for index in range(10):
            ret, self.frame = self.cap.read()
        self.frame_preview = cv.resize(
            self.frame, self.frame_preview_size_4resize)
        # self.frame_preview = cv.resize(self.frame, (0,0), fx=0.5, fy=0.5)
        # print('self.frame.shape', self.frame.shape)
        # print('self.frame_preview_size', self.frame_preview_size)
        # print('self.frame_preview.shape', self.frame_preview.shape)

        # init other things
        # print('-')
        self.load_and_prepare_overlay_and_result_image()
        self.prepare_overlay_preview_and_result_image_preview()

        # done
        ##########################################

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
        # print('run... self.frame_preview_size', self.frame_preview_size)
        # print('run... self.frame_preview', self.frame_preview)
        while run:
            # frame_old = frame
            # Capture frame-by-frame
            ret, self.frame = self.cap.read()
            self.frame_preview = cv.resize(
                self.frame, self.frame_preview_size_4resize)

            if self.save_next_frame_flag:
                self.update_result_image_preview()
                cv.imshow(self.WINDOWNAME, self.result_image_preview)
                cv.waitKey(2)
                # now we have updated the gui
                # so we can do the slow things
                self.update_result_image()
                cv.imwrite(self.output_filename_full, self.result_image)
                print("saved as '" + self.output_filename_full + "'")
                self.show_last_saved_frame_flag = True
                self.save_next_frame_flag = False

            # update preview window
            if self.show_last_saved_frame_flag:
                cv.imshow(self.WINDOWNAME, self.result_image)
                # the & 0xFF is needed for 64-bit machines
                key = cv.waitKey(1000) & 0xFF
                self.show_last_saved_frame_flag = False
            else:
                if self.show_preview:
                    cv.imshow(self.WINDOWNAME, self.frame_preview)
                else:
                    temp = self.frame.copy()
                    cv.putText(
                        temp,
                        'FULL RESOLUTION ACTIVE',
                        (10, 50),
                        cv.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
                    cv.imshow(self.WINDOWNAME, temp)
                # the & 0xFF is needed for 64-bit machines
                key = cv.waitKey(1) & 0xFF

            run = not self.check_exit_key(key)
            if key == ord('s'):
                self.save_next_frame()
            elif key == ord('f'):
                self.toggle_fullscreen()
            elif key == ord('r'):
                self.toggle_preview()

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

    def toggle_preview(self):
        """Toggle preview."""
        self.show_preview = not self.show_preview
        print('toggle show_preview to', self.show_preview)

    def load_and_prepare_overlay_and_result_image(self):
        """Load overlay image."""
        self.overlay_img = cv.imread(
            self.overlay_filename, cv.IMREAD_UNCHANGED)
        self.overlay_size = self.overlay_img.shape[:2]
        # check if we need to move the frame or the overlay...
        # negative overlay positions meaning we need to move the frame.
        frame_pos_height, frame_pos_width = (0, 0)
        overlay_pos_height, overlay_pos_width = self.overlay_position
        if self.overlay_position[0] < 0:
            frame_pos_height = self.overlay_position[0] * -1
            overlay_pos_height = 0
        if self.overlay_position[1] < 0:
            frame_pos_width = self.overlay_position[1] * -1
            overlay_pos_width = 0
        self.overlay_position = (overlay_pos_height, overlay_pos_width)
        self.frame_position = (frame_pos_height, frame_pos_width)
        # calculate result image size
        #
        # TODO(s-light): fix this calculation..
        # for now we just assume the overlay is bigger ;-)
        self.result_image_size = self.overlay_size
        #
        # overlay_height, overlay_width = self.overlay_size
        # result_height, result_width = self.frame_size
        # check what is bigger - frame or overlay
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

        self.reset_result_image()

    def prepare_overlay_preview_and_result_image_preview(self):
        """Load overlay image."""
        # now calculate preview versions
        self.overlay_preview_size = (
            (
                self.overlay_size[0]
                * self.frame_preview_size[0]
                // self.frame_size[0]
            ),
            (
                self.overlay_size[1]
                * self.frame_preview_size[1]
                // self.frame_size[1]
            ),
        )
        self.result_image_preview_size = (
            (
                self.result_image_size[0]
                * self.frame_preview_size[0]
                // self.frame_size[0]
            ),
            (
                self.result_image_size[1]
                * self.frame_preview_size[1]
                // self.frame_size[1]
            ),
        )
        self.frame_preview_position = (
            (
                self.frame_position[0]
                * self.frame_preview_size[0]
                // self.frame_size[0]
            ),
            (
                self.frame_position[1]
                * self.frame_preview_size[1]
                // self.frame_size[1]
            ),
        )
        self.overlay_preview_position = (
            (
                self.overlay_position[0]
                * self.frame_preview_size[0]
                // self.frame_size[0]
            ),
            (
                self.overlay_position[1]
                * self.frame_preview_size[1]
                // self.frame_size[1]
            ),
        )
        self.overlay_preview = cv.resize(
            self.overlay_img,
            (self.overlay_preview_size[1], self.overlay_preview_size[0]))
        self.reset_result_image_preview()

        # print('debug output')
        # print('self.overlay_img.shape[:2]', self.overlay_img.shape[:2])
        # print('size[0] == height')
        # print('size[1] == width')
        # # For a regular image image.shape[] will give you
        # # height, width and channels, in that order.
        # print('----')
        # print('frame_size', self.frame_size)
        # print('overlay_size', self.overlay_size)
        # print('result_image_size', self.result_image_size)
        # print('frame_position', self.frame_position)
        # print('overlay_position', self.overlay_position)
        # print('----')
        # print('frame_preview_size', self.frame_preview_size)
        # print('overlay_preview_size', self.overlay_preview_size)
        # print('result_image_preview_size', self.result_image_preview_size)
        # print('frame_preview_position', self.frame_preview_position)
        # print('overlay_preview_position', self.overlay_preview_position)
        # print('----')
        # print('show overlay_preview')
        # cv.imshow(self.WINDOWNAME, self.overlay_preview)
        # cv.waitKey(2000)
        # print('show result_image_preview')
        # self.update_result_image_preview()
        # cv.imshow(self.WINDOWNAME, self.result_image_preview)
        # cv.waitKey(5000)
        # print('----')
        # print('done.')

    def init_frame_buffers(self):
        """Init frame buffers to white."""
        n_channels = 3
        height, width = self.frame_size
        shape = (height, width, n_channels)
        self.frame = np.full(shape, (0, 200, 255), dtype=np.uint8)
        # cv.imshow(self.WINDOWNAME, self.frame)
        # cv.waitKey(1000)
        height, width = self.frame_preview_size
        shape = (height, width, n_channels)
        self.frame_preview = np.full(shape, (255, 200, 0), dtype=np.uint8)
        # cv.imshow(self.WINDOWNAME, self.frame_preview)
        # cv.waitKey(1000)

    def reset_result_image(self):
        """Reset result_image to transparent."""
        # create transparent image
        # thanks for the example from
        # https://stackoverflow.com/a/44595221/574981
        # RGBA == 4
        n_channels = 4
        height, width = self.result_image_size
        shape = (height, width, n_channels)
        # init to black
        # self.result_image = np.zeros(shape, dtype=np.uint8)
        # init to white but fully transparent
        self.result_image = np.full(shape, (255, 255, 255, 0), dtype=np.uint8)

    def update_result_image(self):
        """Update result image."""
        self.reset_result_image()
        frame = cv.cvtColor(self.frame, cv.COLOR_RGB2RGBA).copy()
        self.overlay_image_alpha(
            self.result_image,
            frame,
            self.frame_position)
        # self.result_image[
        #     self.frame_position[1]:(self.frame_position[1]+1080),
        #     self.frame_position[0]:(self.frame_position[0]+1920)
        # ] = frame[0:frame.shape[0], 0:frame.shape[1]]
        self.overlay_image_alpha(
            self.result_image,
            self.overlay_img,
            self.overlay_position)

    # fast preview handling
    def reset_result_image_preview(self):
        """Reset result_image to transparent."""
        # create transparent image
        # thanks for the example from
        # https://stackoverflow.com/a/44595221/574981
        # RGBA == 4
        n_channels = 4
        height, width = self.result_image_preview_size
        shape = (height, width, n_channels)
        # init to black
        # self.result_image = np.zeros(shape, dtype=np.uint8)
        # init to white but fully transparent
        self.result_image_preview = np.full(
            shape, (255, 255, 255, 0), dtype=np.uint8)

    def update_result_image_preview(self):
        """Update result preview image."""
        self.reset_result_image_preview()
        frame = cv.cvtColor(self.frame_preview, cv.COLOR_RGB2RGBA).copy()
        self.overlay_image_alpha(
            self.result_image_preview,
            frame,
            self.frame_preview_position)
        # self.result_image_preview[
        #     self.frame_preview_position[0]:self.frame_preview_position[1],
        #     frame.shape[0]:frame.shape[1]
        # ] = frame
        self.overlay_image_alpha(
            self.result_image_preview,
            self.overlay_preview,
            self.overlay_preview_position)

    # helper function
    def overlay_image_alpha(self, img, img_overlay, pos):
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
    print('sys.path: ' + sys.path)
    print(42*'*')

    # print(42*'*')
    # print('debug info')
    # events = [i for i in dir(cv) if 'EVENT' in i]
    # print(events)
    # print(42*'*')

    cam = '/dev/video_cam_C1'
    output_filename_default = "./captured/test_{date_part}.png"
    overlay_filename_default = "./overlay/picture_frame.png"

    parser = argparse.ArgumentParser(
        description="test cam."
    )

    parser.add_argument(
        "-d",
        "--device",
        help="specify camera device number to use. (defaults to {})".format(
            cam
        ),
        metavar='DEVICE',
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

    print('device', args.device)
    print('output_filename', args.output_filename)
    print('output_filename', args.overlay_filename)
    cam = Cam(
        camera_device=args.device,
        output_filename_template=args.output_filename,
        overlay_filename=args.overlay_filename,
        fullscreen=args.fullscreen
    )
    cam.start()
    cam.run()


if __name__ == "__main__":
    main()
