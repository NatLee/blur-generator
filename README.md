# Blur Generator

[![PyPI pyversions](https://img.shields.io/pypi/pyversions/BlurGenerator.svg)](https://pypi.python.org/pypi/BlurGenerator/)[![PyPI implementation](https://img.shields.io/pypi/implementation/BlurGenerator.svg)](https://pypi.python.org/pypi/BlurGenerator/)

[![Test](https://github.com/NatLee/blur-generator/actions/workflows/test.yml/badge.svg)](https://github.com/NatLee/blur-generator/actions/workflows/test.yml)[![Release](https://github.com/NatLee/blur-generator/actions/workflows/release.yml/badge.svg)](https://github.com/NatLee/blur-generator/actions/workflows/release.yml)

[![PyPI status](https://img.shields.io/pypi/status/BlurGenerator.svg)](https://pypi.python.org/pypi/BlurGenerator/)[![PyPI license](https://img.shields.io/pypi/l/BlurGenerator.svg)](https://pypi.python.org/pypi/BlurGenerator/)

[![PyPI version fury.io](https://badge.fury.io/py/BlurGenerator.svg)](https://pypi.python.org/pypi/BlurGenerator/)

[![PyPI download month](https://img.shields.io/pypi/dm/BlurGenerator.svg)](https://pypi.python.org/pypi/BlurGenerator/)[![PyPI download week](https://img.shields.io/pypi/dw/BlurGenerator.svg)](https://pypi.python.org/pypi/BlurGenerator/)[![PyPI download day](https://img.shields.io/pypi/dd/BlurGenerator.svg)](https://pypi.python.org/pypi/BlurGenerator/)

[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

This tool is for generating blur on images.

There are 3 types of blur modes of `motion`, `lens`, or `gaussian`.

We can use the results on model training or something else.

> You can find a simply use case with deep learning in https://github.com/NatLee/simply-blur-detector

## Installation

```bash
pip install blurgenerator
```

Check it on [PyPI](https://pypi.org/project/BlurGenerator/).

If `import cv2` fails with an error like `libxcb.so.1: cannot open shared object file` or `libGL.so.1: cannot open shared object file` (common in slim Docker images or on servers without a desktop), install the system libraries OpenCV needs, for example on Debian/Ubuntu:

```bash
apt-get install -y libgl1 libglib2.0-0
```

## Usage

```bash
blurgenerator --help
```

```text
usage: blurgenerator [-h] [--input INPUT] [--input_depth_map INPUT_DEPTH_MAP]
                     [--output OUTPUT] [--type TYPE]
                     [--motion_blur_size MOTION_BLUR_SIZE]
                     [--motion_blur_angle MOTION_BLUR_ANGLE]
                     [--lens_radius LENS_RADIUS]
                     [--lens_components LENS_COMPONENTS]
                     [--lens_exposure_gamma LENS_EXPOSURE_GAMMA]
                     [--gaussian_kernel GAUSSIAN_KERNEL]
                     [--depth_num_layers DEPTH_NUM_LAYERS]
                     [--depth_min_blur DEPTH_MIN_BLUR]
                     [--depth_max_blur DEPTH_MAX_BLUR]

options:
  -h, --help            show this help message and exit
  --input INPUT         Specific path of image as `input`.
  --input_depth_map INPUT_DEPTH_MAP
                        Specific path of depth image as `input_depth_map`.
  --output OUTPUT       Specific path for `output`. Default is `./result.png`.
  --type TYPE           Blur type of `motion`, `lens`, or `gaussian`. Default
                        is `motion`.
  --motion_blur_size MOTION_BLUR_SIZE
                        Size for motion blur. Default is 100.
  --motion_blur_angle MOTION_BLUR_ANGLE
                        Angle for motion blur. Default is 30.
  --lens_radius LENS_RADIUS
                        Radius for lens blur. Default is 5.0.
  --lens_components LENS_COMPONENTS
                        Components for lens blur. Default is 4.
  --lens_exposure_gamma LENS_EXPOSURE_GAMMA
                        Exposure gamma for lens blur. Default is 2.
  --gaussian_kernel GAUSSIAN_KERNEL
                        Kernel for gaussian. Default is 100.
  --depth_num_layers DEPTH_NUM_LAYERS
                        Layer for depth blur. Default is 10.
  --depth_min_blur DEPTH_MIN_BLUR
                        Min. blur for depth blur. Default is 1.
  --depth_max_blur DEPTH_MAX_BLUR
                        Max. blur for depth blur. Default is 100.
```

## Example and Result

The examples below use the sample images in [`doc/`](https://github.com/NatLee/blur-generator/tree/main/doc). Run them from the root of this repository, or replace the paths with your own images.

### Common use

- Original image

![original image](https://github.com/NatLee/blur-generator/raw/main/doc/test.png)

#### Usage

- Motion blur

`blurgenerator --type motion --input ./doc/test.png --output ./doc/motion.png`

```python
import cv2
from blurgenerator import motion_blur
img = cv2.imread('./doc/test.png')
result = motion_blur(img, size=100, angle=30)
cv2.imwrite('./output.png', result)
```

![motion blur image](https://github.com/NatLee/blur-generator/raw/main/doc/motion.png)

- Lens blur

`blurgenerator --type lens --input ./doc/test.png --output ./doc/lens.png`

```python
import cv2
from blurgenerator import lens_blur
img = cv2.imread('./doc/test.png')
result = lens_blur(img, radius=5, components=4, exposure_gamma=2)
cv2.imwrite('./output.png', result)
```

![lens blur image](https://github.com/NatLee/blur-generator/raw/main/doc/lens.png)

- Gaussian blur

`blurgenerator --type gaussian --input ./doc/test.png --output ./doc/gaussian.png`

```python
import cv2
from blurgenerator import gaussian_blur
img = cv2.imread('./doc/test.png')
result = gaussian_blur(img, 100)
cv2.imwrite('./output.png', result)
```

![gaussian blur image](https://github.com/NatLee/blur-generator/raw/main/doc/gaussian.png)

### With depth map

Feature from this [issue](https://github.com/NatLee/blur-generator/issues/1).

- Original image

![photo](https://github.com/NatLee/blur-generator/raw/main/doc/depth-test.jpg)

- Depth map

![depth map](https://github.com/NatLee/blur-generator/raw/main/doc/depth-map-test.png)

#### Usage

- Motion blur with depth map

`blurgenerator --input ./doc/depth-test.jpg --type motion --input_depth_map ./doc/depth-map-test.png --depth_num_layers 5 --depth_min_blur 1 --depth_max_blur 50 --output ./doc/depth-motion-output.png`

```python
import cv2
from blurgenerator import motion_blur_with_depth_map
img = cv2.imread('./doc/depth-test.jpg')
depth_img = cv2.imread('./doc/depth-map-test.png')
result = motion_blur_with_depth_map(
   img,
   depth_map=depth_img,
   angle=30,
   num_layers=10,
   min_blur=1,
   max_blur=50
)
cv2.imwrite('./output.png', result)
```

![depth motion blur image](https://github.com/NatLee/blur-generator/raw/main/doc/depth-motion-output.png)

- Lens blur with depth map

`blurgenerator --input ./doc/depth-test.jpg --type lens --input_depth_map ./doc/depth-map-test.png --depth_num_layers 3 --depth_min_blur 1 --depth_max_blur 50 --output ./doc/depth-lens-output.png`

```python
import cv2
from blurgenerator import lens_blur_with_depth_map
img = cv2.imread('./doc/depth-test.jpg')
depth_img = cv2.imread('./doc/depth-map-test.png')
result = lens_blur_with_depth_map(
   img,
   depth_map=depth_img,
   components=5,
   exposure_gamma=5,
   num_layers=10,
   min_blur=1,
   max_blur=50
)
cv2.imwrite('./output.png', result)
```

![depth lens blur image](https://github.com/NatLee/blur-generator/raw/main/doc/depth-lens-output.png)

- Gaussian blur with depth map

`blurgenerator --input ./doc/depth-test.jpg --type gaussian --input_depth_map ./doc/depth-map-test.png --depth_num_layers 3 --depth_min_blur 1 --depth_max_blur 50 --output ./doc/depth-gaussian-output.png`

```python
import cv2
from blurgenerator import gaussian_blur_with_depth_map
img = cv2.imread('./doc/depth-test.jpg')
depth_img = cv2.imread('./doc/depth-map-test.png')
result = gaussian_blur_with_depth_map(
   img,
   depth_map=depth_img,
   sigma=5,
   num_layers=10,
   min_blur=1,
   max_blur=50
)
cv2.imwrite('./output.png', result)
```

![depth gaussian blur image](https://github.com/NatLee/blur-generator/raw/main/doc/depth-gaussian-output.png)

## Development

Run the tests (Python 3.7 or newer):

```bash
pip install -e . pytest
pytest tests
```

`tox -e py` runs the same tests plus a check that the source package contains every file.

## Contributor

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center"><a href="https://github.com/NatLee"><img src="https://avatars.githubusercontent.com/u/10178964?v=3?s=100" width="100px;" alt="Nat Lee"/><br /><sub><b>Nat Lee</b></sub></a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

## Reference

- [Circularly symmetric convolution and lens blur](http://yehar.com/blog/?p=1495)


## LICENSE

[MIT](LICENSE)
