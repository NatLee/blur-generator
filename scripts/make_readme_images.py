"""Regenerate the images in doc/readme/ that the README shows.

Run it from the root of the repository after installing the package (``pip install -e .``):

    python scripts/make_readme_images.py

Every picture is real output of blurgenerator. The commands below are the ones printed in the README;
they run on the sample images in doc/ and the results are resized and saved as small JPEGs for the page.
"""
import os
import shlex
import subprocess
import tempfile

import cv2
import numpy as np

from blurgenerator import lens_blur_with_depth_map

OUT_DIR = os.path.join('doc', 'readme')
PHOTO = os.path.join('doc', 'depth-test.jpg')
DEPTH_MAP = os.path.join('doc', 'depth-map-test.png')
SAMPLE = os.path.join(OUT_DIR, 'sample.jpg')

# README "Gallery": one command per blur type on doc/readme/sample.jpg.
GALLERY = {
    'gallery-motion.jpg': 'blurgenerator --type motion --motion_blur_size 25 '
                          '--input ./doc/readme/sample.jpg --output motion.jpg',
    'gallery-lens.jpg': 'blurgenerator --type lens --lens_radius 8 --lens_exposure_gamma 5 '
                        '--input ./doc/readme/sample.jpg --output lens.jpg',
    'gallery-gaussian.jpg': 'blurgenerator --type gaussian --gaussian_kernel 31 '
                            '--input ./doc/readme/sample.jpg --output gaussian.jpg',
}

# README "Blur by depth" and "Focus on a depth": the full-size photo and depth map in doc/.
DEPTH = {
    'depth-motion.jpg': 'blurgenerator --type motion --input ./doc/depth-test.jpg '
                        '--input_depth_map ./doc/depth-map-test.png --depth_max_blur 50 --output depth-motion.png',
    'depth-lens.jpg': 'blurgenerator --type lens --input ./doc/depth-test.jpg '
                      '--input_depth_map ./doc/depth-map-test.png --depth_max_blur 40 --output depth-lens.png',
    'depth-gaussian.jpg': 'blurgenerator --type gaussian --input ./doc/depth-test.jpg '
                          '--input_depth_map ./doc/depth-map-test.png --depth_max_blur 50 '
                          '--output depth-gaussian.png',
    'focus-rocks.jpg': 'blurgenerator --type lens --input ./doc/depth-test.jpg '
                       '--input_depth_map ./doc/depth-map-test.png --depth_max_blur 40 '
                       '--depth_focus_point 480 1150 --depth_focus_range 20 --output focus.png',
}

DISPLAY_WIDTH = 512


def save_jpeg(name, img, width=None):
    if width is not None and img.shape[1] != width:
        height = round(img.shape[0] * width / img.shape[1])
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    path = os.path.join(OUT_DIR, name)
    assert cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 85]), path
    print(f'{path:36s} {img.shape[1]}x{img.shape[0]}  {os.path.getsize(path) // 1024} KB')


def run(command, tmp):
    """Run a README command, but write its output into ``tmp``; return the image it wrote."""
    args = shlex.split(command)
    index = args.index('--output') + 1
    args[index] = os.path.join(tmp, os.path.basename(args[index]))
    subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
    return cv2.imread(args[index])


def checkerboard_strips():
    """Three strips for the README: a depth gradient, the default depth blur and a focused one."""
    height, width = 64, 768
    yy, xx = np.mgrid[:height, :width]
    checker = (((yy // 16) + (xx // 16)) % 2 * 200 + 30).astype(np.uint8)
    img = cv2.merge([checker] * 3)
    depth = cv2.merge([np.tile(np.linspace(0, 255, width).astype(np.uint8), (height, 1))] * 3)
    settings = dict(components=4, exposure_gamma=1, num_layers=12, min_blur=1, max_blur=12)
    default = lens_blur_with_depth_map(img, depth, **settings)
    focused = lens_blur_with_depth_map(img, depth, focus=128, focus_range=40, **settings)
    for name, strip in (('strip-depth.png', depth), ('strip-default.png', default), ('strip-focus.png', focused)):
        path = os.path.join(OUT_DIR, name)
        assert cv2.imwrite(path, strip, [cv2.IMWRITE_PNG_COMPRESSION, 9]), path
        print(f'{path:36s} {width}x{height}  {os.path.getsize(path) // 1024} KB')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    photo = cv2.imread(PHOTO)

    # The gallery sample: the pavilion and the sunlit floor, 900x600 pixels scaled down to 600x400.
    save_jpeg('sample.jpg', cv2.resize(photo[600:1200, 800:1700], (600, 400), interpolation=cv2.INTER_AREA))
    save_jpeg('depth-photo.jpg', photo, DISPLAY_WIDTH)
    save_jpeg('depth-map.jpg', cv2.imread(DEPTH_MAP), DISPLAY_WIDTH)

    with tempfile.TemporaryDirectory() as tmp:
        for name, command in GALLERY.items():
            save_jpeg(name, run(command, tmp))
        for name, command in DEPTH.items():
            save_jpeg(name, run(command, tmp), DISPLAY_WIDTH)

    checkerboard_strips()


if __name__ == '__main__':
    main()
