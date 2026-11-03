"""
Command line interface: ``blurgenerator --type lens --input photo.jpg --output result.png``.

Exit codes: the checks that print a ``-----`` message on stdout (missing input, wrong file type, unknown
blur type, output not written) end with exit code 0, as they always have. Errors that used to end in a
Python traceback (an unreadable image, an invalid blur setting) now print a message on stderr instead and
still exit with code 1.
"""
import argparse
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple, NoReturn, Optional, Sequence

import cv2
import numpy as np

from .depth import gaussian_blur_with_depth_map, lens_blur_with_depth_map, motion_blur_with_depth_map
from .gaussian_blur import DEFAULT_SIGMA, gaussian_blur
from .lens_blur import lens_blur
from .motion_blur import motion_blur


class _Blur(NamedTuple):
    """One ``--type``: how to run it without and with a depth map."""
    #: The option that sets the strength without a depth map. With one, the depth map decides.
    amount_option: str
    plain: Callable[[argparse.Namespace, np.ndarray], np.ndarray]
    with_depth: Callable[..., np.ndarray]


#: Every --type, in the order the help lists them. A new blur type is a new entry here.
BLURS = {
    'motion': _Blur(
        'motion_blur_size',
        lambda args, img: motion_blur(img, size=args.motion_blur_size, angle=args.motion_blur_angle),
        lambda args, img, depth_map, **depth: motion_blur_with_depth_map(
            img, depth_map, angle=args.motion_blur_angle, **depth),
    ),
    'lens': _Blur(
        'lens_radius',
        lambda args, img: lens_blur(img, radius=args.lens_radius, components=args.lens_components,
                                    exposure_gamma=args.lens_exposure_gamma),
        lambda args, img, depth_map, **depth: lens_blur_with_depth_map(
            img, depth_map, components=args.lens_components, exposure_gamma=args.lens_exposure_gamma, **depth),
    ),
    'gaussian': _Blur(
        'gaussian_kernel',
        # Both use the library's default sigma.
        lambda args, img: gaussian_blur(img, args.gaussian_kernel),
        lambda args, img, depth_map, **depth: gaussian_blur_with_depth_map(img, depth_map, **depth),
    ),
}
BLUR_TYPES = tuple(BLURS)
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')
#: The depth-of-field options; they only work together with --input_depth_map.
FOCUS_OPTIONS = ('depth_focus', 'depth_focus_range', 'depth_focus_point')


def _listing(words: Sequence[str], conjunction: str) -> str:
    """The words as in a sentence: ``a, b or c``."""
    return f'{", ".join(words[:-1])} {conjunction} {words[-1]}' if len(words) > 1 else words[0]


def _quoted(words: Sequence[str]) -> List[str]:
    return [f'`{word}`' for word in words]


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the ``blurgenerator`` command."""
    parser = argparse.ArgumentParser(
        prog='blurgenerator',
        description='Add motion, lens or gaussian blur to an image, optionally guided by a depth map.',
    )

    parser.add_argument('--input', type=str, default=None,
                        help=f'Image to blur ({_listing(IMAGE_SUFFIXES, "or")}). Required.')
    parser.add_argument('--input_depth_map', type=str, default=None,
                        help=f'Depth map ({_listing(IMAGE_SUFFIXES, "or")}, same size as the input) that decides '
                             'how much each part of the image is blurred. Optional.')

    parser.add_argument('--output', type=str, default='./result.png',
                        help='Where to save the result. Default is `%(default)s`.')

    parser.add_argument('--type', type=str, default='motion',
                        help=f'Blur type: {_listing(_quoted(BLUR_TYPES), "or")}. Default is `%(default)s`.')

    parser.add_argument('--motion_blur_size', type=int, default=100,
                        help='Length of the motion blur in pixels. Default is %(default)s.')
    parser.add_argument('--motion_blur_angle', type=int, default=30,
                        help='Direction of the motion blur in degrees. Default is %(default)s.')

    parser.add_argument('--lens_radius', type=float, default=5.0,
                        help='Radius of the lens blur in pixels. Default is %(default)s.')
    parser.add_argument('--lens_components', type=int, default=4,
                        help='Kernel components of the lens blur, 1 to 6; more is rounder and slower. '
                             'Default is %(default)s.')
    parser.add_argument('--lens_exposure_gamma', type=int, default=2,
                        help='How much bright spots bloom in the lens blur; 1 turns it off. Default is %(default)s.')

    parser.add_argument('--gaussian_kernel', type=int, default=100,
                        help=f'Kernel size of the gaussian blur in pixels (sigma is {DEFAULT_SIGMA}). '
                             'Default is %(default)s.')

    # depth blur settings
    parser.add_argument('--depth_num_layers', type=int, default=10,
                        help='Number of depth layers; more layers give smoother transitions but take longer. '
                             'Default is %(default)s.')
    parser.add_argument('--depth_min_blur', type=int, default=1,
                        help='Blur at depth 0, or right next to the focus. Default is %(default)s.')
    parser.add_argument('--depth_max_blur', type=int, default=100,
                        help='Blur at depth 255, or farthest from the focus. Default is %(default)s.')

    # depth of field: focus on one depth, blur grows in both directions (issue #5)
    parser.add_argument('--depth_focus', type=int, default=None,
                        help='Depth (0-255) to keep sharp; the blur then grows with the distance from it, '
                             'towards both 0 and 255. Default is off: the blur grows with the depth.')
    parser.add_argument('--depth_focus_range', type=int, default=0,
                        help='Width of the depth band around the focus that stays sharp: 10 keeps '
                             'focus-5 to focus+5 sharp. Default is %(default)s.')
    parser.add_argument('--depth_focus_point', type=int, nargs=2, metavar=('X', 'Y'), default=None,
                        help='Focus on the depth of this pixel of the depth map (X = column, Y = row), '
                             'instead of --depth_focus.')

    return parser


def _focus_options_given(parser: argparse.ArgumentParser, args: argparse.Namespace) -> bool:
    return any(getattr(args, option) != parser.get_default(option) for option in FOCUS_OPTIONS)


def _abort(message: str) -> NoReturn:
    """Print ``message`` on stderr and exit with code 1."""
    sys.exit(f'----- {message}')


def _is_supported_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def _read_image(path: Path, option: str) -> np.ndarray:
    image = cv2.imread(path.absolute().as_posix())
    if image is None:
        _abort(f'Could not read `{path}` ({option}) as an image.')
    return image


def _write_image(path: str, image: np.ndarray) -> None:
    try:
        written = cv2.imwrite(path, image)
    except cv2.error:
        _abort(f'Could not write `{path}`: the file name must end in an image extension such as .png or .jpg.')
    if not written:
        print(f'----- Could not write `{path}`. Check that the folder exists.')


def _depth_blur(args: argparse.Namespace, img: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """The blur selected by ``--type`` with a depth map; the same functions as the Python API."""
    return BLURS[args.type].with_depth(
        args, img, depth_map,
        num_layers=args.depth_num_layers,
        min_blur=args.depth_min_blur,
        max_blur=args.depth_max_blur,
        focus=args.depth_focus,
        focus_range=args.depth_focus_range,
        focus_point=args.depth_focus_point,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point of the ``blurgenerator`` command. ``argv`` defaults to the command line arguments."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.input:
        print('----- Please specify an input image with `--input`.')
        return

    img_path = Path(args.input)

    if not img_path.is_file():
        print(f'----- `{args.input}` is not a file.')
        return

    if not _is_supported_image(img_path):
        print(f'----- Only {_listing(IMAGE_SUFFIXES, "and")} images are supported.')
        return

    if args.type not in BLURS:
        print(f'----- Unknown blur type `{args.type}`. Use {_listing(_quoted(BLUR_TYPES), "or")}.')
        return

    img = _read_image(img_path, '--input')

    if args.input_depth_map is None:
        print(f'----- Generating `{args.type}` blur.')
        if _focus_options_given(parser, args):
            print(f'----- {_listing(_quoted(FOCUS_OPTIONS), "and")} will be ignored without `--input_depth_map`.')
        try:
            result = BLURS[args.type].plain(args, img)
        except ValueError as error:
            _abort(str(error))
        _write_image(args.output, result)
        return

    depth_map_path = Path(args.input_depth_map)
    if not depth_map_path.is_file():
        print(f'----- `{args.input_depth_map}` (--input_depth_map) is not a file.')
        return
    if not _is_supported_image(depth_map_path):
        print(f'----- Only {_listing(IMAGE_SUFFIXES, "and")} images are supported.')
        return

    print(f'----- Generating `{args.type}` blur with depth map.')
    for blur in BLURS.values():
        print(f'----- `{blur.amount_option}` will be ignored.')

    depth_map = _read_image(depth_map_path, '--input_depth_map')
    try:
        result = _depth_blur(args, img, depth_map)
    except ValueError as error:
        _abort(str(error))
    _write_image(args.output, result)
