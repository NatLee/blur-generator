"""
Command line interface: ``blurgenerator --type lens --input photo.jpg --output result.png``.
"""
import argparse
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from .depth import gaussian_blur_with_depth_map, lens_blur_with_depth_map, motion_blur_with_depth_map
from .gaussian_blur import gaussian_blur
from .lens_blur import lens_blur
from .motion_blur import motion_blur

BLUR_TYPES = ('motion', 'lens', 'gaussian')
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``blurgenerator`` command."""
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default=None, help='Specific path of image as `input`.')
    parser.add_argument('--input_depth_map', type=str, default=None, help='Specific path of depth image as `input_depth_map`.')

    parser.add_argument('--output', type=str, default='./result.png', help='Specific path for `output`. Default is `./result.png`.')

    parser.add_argument('--type', type=str, default='motion', help='Blur type of `motion`, `lens`, or `gaussian`. Default is `motion`.')

    parser.add_argument('--motion_blur_size', type=int, default=100, help='Size for motion blur. Default is 100.')
    parser.add_argument('--motion_blur_angle', type=int, default=30, help='Angle for motion blur. Default is 30.')

    parser.add_argument('--lens_radius', type=float, default=5.0, help='Radius for lens blur. Default is 5.0.')
    parser.add_argument('--lens_components', type=int, default=4, help='Components for lens blur. Default is 4.')
    parser.add_argument('--lens_exposure_gamma', type=int, default=2, help='Exposure gamma for lens blur. Default is 2.')

    parser.add_argument('--gaussian_kernel', type=int, default=100, help='Kernel for gaussian. Default is 100.')

    # depth blur settings
    parser.add_argument('--depth_num_layers', type=int, default=10, help='Layer for depth blur. Default is 10.')
    parser.add_argument('--depth_min_blur', type=int, default=1, help='Min. blur for depth blur. Default is 1.')
    parser.add_argument('--depth_max_blur', type=int, default=100, help='Max. blur for depth blur. Default is 100.')

    return parser


def _is_supported_image(path: Path) -> bool:
    return path.suffix in IMAGE_SUFFIXES


def _plain_blur(args: argparse.Namespace) -> Callable[[np.ndarray], np.ndarray]:
    """The blur selected by ``--type``, with its own options, for use without a depth map."""
    if args.type == 'motion':
        return lambda img: motion_blur(img, size=args.motion_blur_size, angle=args.motion_blur_angle)
    if args.type == 'lens':
        return lambda img: lens_blur(img, radius=args.lens_radius, components=args.lens_components,
                                     exposure_gamma=args.lens_exposure_gamma)
    return lambda img: gaussian_blur(img, args.gaussian_kernel)


def _depth_blur(args: argparse.Namespace, img: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """The blur selected by ``--type`` with a depth map; the same functions as the Python API."""
    depth_args = dict(
        num_layers=args.depth_num_layers,
        min_blur=args.depth_min_blur,
        max_blur=args.depth_max_blur,
    )
    if args.type == 'motion':
        return motion_blur_with_depth_map(img, depth_map, angle=args.motion_blur_angle, **depth_args)
    if args.type == 'lens':
        return lens_blur_with_depth_map(img, depth_map, components=args.lens_components,
                                        exposure_gamma=args.lens_exposure_gamma, **depth_args)
    # gaussian_blur() defaults to sigma=5, which the CLI has always used.
    return gaussian_blur_with_depth_map(img, depth_map, sigma=5, **depth_args)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of the ``blurgenerator`` command. ``argv`` defaults to the command line arguments."""
    args = build_parser().parse_args(argv)

    if not args.input:
        print('----- Please specific image for input.')
        return

    img_path = Path(args.input)

    if not img_path.is_file():
        print('----- `img_path` is not a file!')
        return

    if not _is_supported_image(img_path):
        print('----- Only support common types of image `.jpg` and `.png`.')
        return

    if args.type not in BLUR_TYPES:
        print('----- No type has been selected. Please specific `motion`, `lens`, or `gaussian`.')
        return

    img = cv2.imread(img_path.absolute().as_posix())

    if args.input_depth_map is None:
        print(f'----- Generating `{args.type}` blur.')
        cv2.imwrite(args.output, _plain_blur(args)(img))
        return

    depth_map_path = Path(args.input_depth_map)
    if not depth_map_path.is_file():
        print('----- `input_depth_map` is not a file!')
        return
    if not _is_supported_image(depth_map_path):
        print('----- Only support common types of image `.jpg` and `.png`.')
        return

    print(f'----- Generating `{args.type}` blur with depth map.')
    print('----- `motion_blur_size` will be ignored.')
    print('----- `lens_radius` will be ignored.')
    print('----- `gaussian_kernel` will be ignored.')

    depth_map = cv2.imread(depth_map_path.absolute().as_posix())
    cv2.imwrite(args.output, _depth_blur(args, img, depth_map))
