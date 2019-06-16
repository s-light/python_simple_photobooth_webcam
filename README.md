<!--lint disable list-item-indent-->
<!--lint disable list-item-bullet-indent-->

# python_simple_photobooth_webcam
simple photobooth script for a webcam usage..

cam.py works as main photobooth script for now ;-)

## Install

### Prerequisites
- [python3](https://www.python.org/downloads/)
- installed [OpenCV with python support](https://docs.opencv.org/master/da/df6/tutorial_py_table_of_contents_setup.html)
- usb UVC FullHD compatible webcam.
    i am using [this nice device](https://www.kurokesu.com/shop/cameras/CAMUSB1) ;-)
    it *should* run with every modern FullHD webcam..

- clone or [download](https://github.com/s-light/python_simple_photobooth_webcam/archive/master.zip) and extract this repository to your computer
```bash
git clone https://github.com/s-light/python_simple_photobooth_webcam.git
```
- in the terminal / command prompt change to the folder
`cd python_simple_photobooth_webcam`

## Usage
type `./cam.py --help` or `python3 cam.py --help` to show the build in list of arguments
```bash
$ ./cam.py --help
usage: cam.py [-h] [-d DEVICE] [-o OUTPUT_FILENAME] [-w OVERLAY_FILENAME] [-f]

test cam.

optional arguments:
  -h, --help            show this help message and exit
  -d DEVICE, --device DEVICE
                        specify camera device number to use. (defaults to
                        /dev/video_cam_C1)
  -o OUTPUT_FILENAME, --output_filename OUTPUT_FILENAME
                        specify a location for the output file (defaults to
                        ./captured/test_{date_part}.png)
  -w OVERLAY_FILENAME, --overlay_filename OVERLAY_FILENAME
                        specify a location for the overlay file (defaults to
                        ./overlay/picture_frame.png)
  -f, --fullscreen      start in fullscreen mode

```

most likely you want to minimally use something like this:
```bash
./cam.py --device="/dev/video0"
```
for windows try with  `--device="0"` → this should use the first/default cam.
(not tested)


## ToDo

... much things can be improved:
- overlay/frame position is hard-coded
- overlay/frame can only be bigger than camera image.
- cam resolution is hard-coded
- extract all the options to `photobooth_config.json`
