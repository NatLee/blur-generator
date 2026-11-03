"""
Blur Generator: add motion, lens or Gaussian blur to images, optionally driven by a depth map.
"""
__version__ = "1.1.0"

from .motion_blur import motion_blur
from .lens_blur import lens_blur
from .gaussian_blur import gaussian_blur

from .depth import motion_blur_with_depth_map
from .depth import lens_blur_with_depth_map
from .depth import gaussian_blur_with_depth_map

from .cli import main

__all__ = [
    'motion_blur',
    'lens_blur',
    'gaussian_blur',
    'motion_blur_with_depth_map',
    'lens_blur_with_depth_map',
    'gaussian_blur_with_depth_map',
    'main',
]
