"""
Blur Maker
"""
import argparse
from pathlib import Path

import cv2

from blurgenerator import motion_blur, lens_blur, gaussian_blur
from blurgenerator.depth import (
    motion_blur_with_depth_map,
    lens_blur_with_depth_map,
    gaussian_blur_with_depth_map,
)

def main():

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
    parser.add_argument('--depth_num_layers', type=int, default=10, help='Layer for depth blur. Default is 3.')
    parser.add_argument('--depth_min_blur', type=int, default=1, help='Min. blur for depth blur. Default is 1.')
    parser.add_argument('--depth_max_blur', type=int, default=100, help='Max. blur for depth blur. Default is 100.')

    # ---------------------------------------------------------------

    args = parser.parse_args()

    if not args.input:
        print('----- Please specific image for input.')
        return

    img_path = Path(args.input)

    if not img_path.is_file():
        print('----- `img_path` is not a file!')
        return

    if img_path.suffix not in ['.jpg', '.jpeg', '.png']:
        print('----- Only support common types of image `.jpg` and `.png`.')
        return

    if args.type not in ['motion', 'lens', 'gaussian']:
        print('----- No type has been selected. Please specific `motion`, `lens`, or `gaussian`.')
        return

    if args.type == 'motion':
        def blur_job(img, size=args.motion_blur_size):
            return motion_blur(img, size=size, angle=args.motion_blur_angle)

    if args.type == 'lens':
        def blur_job(img, radius=args.lens_radius):
            return lens_blur(img, radius=radius, components=args.lens_components, exposure_gamma=args.lens_exposure_gamma)

    if args.type == 'gaussian':
        def blur_job(img, kernel=args.gaussian_kernel):
            return gaussian_blur(img, kernel)

    # ---------------------------------------------------------------

    img = cv2.imread(img_path.absolute().as_posix())

    depth_map_path = args.input_depth_map

    if depth_map_path is None:
        print(f'----- Generating `{args.type}` blur.')
        result = blur_job(img)
        cv2.imwrite(args.output, result)
        return

    depth_map_path = Path(depth_map_path)
    if not depth_map_path.is_file():
        print('----- `input_depth_map` is not a file!')
        return
    if depth_map_path.suffix not in ['.jpg', '.jpeg', '.png']:
        print('----- Only support common types of image `.jpg` and `.png`.')
        return

    print(f'----- Generating `{args.type}` blur with depth map.')
    print('----- `motion_blur_size` will be ignored.')
    print('----- `lens_radius` will be ignored.')
    print('----- `gaussian_kernel` will be ignored.')

    depth_map = cv2.imread(depth_map_path.absolute().as_posix())

    # Same functions as the Python API, so the CLI and the library give the same result.
    depth_args = dict(
        num_layers=args.depth_num_layers,
        min_blur=args.depth_min_blur,
        max_blur=args.depth_max_blur,
    )
    if args.type == 'motion':
        result = motion_blur_with_depth_map(img, depth_map, angle=args.motion_blur_angle, **depth_args)
    elif args.type == 'lens':
        result = lens_blur_with_depth_map(
            img,
            depth_map,
            components=args.lens_components,
            exposure_gamma=args.lens_exposure_gamma,
            **depth_args
        )
    else:
        # gaussian_blur() defaults to sigma=5, which the CLI has always used.
        result = gaussian_blur_with_depth_map(img, depth_map, sigma=5, **depth_args)
    cv2.imwrite(args.output, result)

    return

