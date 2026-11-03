"""The blurgenerator command: every path from the arguments to the written file."""
import os
import re
import unittest

import cv2
import numpy as np

import blurgenerator
from blurgenerator import (
    motion_blur,
    lens_blur,
    gaussian_blur,
    motion_blur_with_depth_map,
    gaussian_blur_with_depth_map,
)
from blurgenerator.cli import build_parser

from .helpers import TempDirTestCase, gradient_depth_map, random_image, run_cli


class TestOptions(unittest.TestCase):

    def test_defaults_are_unchanged(self):
        parser = build_parser()
        expected = {
            'input': None,
            'input_depth_map': None,
            'output': './result.png',
            'type': 'motion',
            'motion_blur_size': 100,
            'motion_blur_angle': 30,
            'lens_radius': 5.0,
            'lens_components': 4,
            'lens_exposure_gamma': 2,
            'gaussian_kernel': 100,
            'depth_num_layers': 10,
            'depth_min_blur': 1,
            'depth_max_blur': 100,
            'depth_focus': None,
            'depth_focus_range': 0,
            'depth_focus_point': None,
        }
        self.assertEqual(vars(parser.parse_args([])), expected)

    def test_help_mentions_every_option_and_its_default(self):
        result = run_cli('--help')
        self.assertEqual(result.code, 0)
        help_text = re.sub(r'\x1b\[[0-9;]*m', '', result.stdout)  # drop colors (Python 3.14+)
        help_text = re.sub(r'\s+', ' ', help_text)
        for option in vars(build_parser().parse_args([])):
            self.assertIn('--' + option, help_text)
        for default in ('`./result.png`', '`motion`', '100', '30', '5.0', '4', '2', '10', '1'):
            self.assertIn('Default is ' + default + '.', help_text)

    def test_package_main_is_the_cli(self):
        self.assertIs(blurgenerator.main, blurgenerator.cli.main)


class TestInputChecks(TempDirTestCase, unittest.TestCase):
    """The checks that print a message and stop without writing anything (exit code 0, as always)."""

    def assert_stops(self, result, message):
        self.assertEqual(result.code, 0)
        self.assertIn(message, result.stdout)
        self.assertFalse(os.path.exists(self.path('out.png')))

    def test_no_input(self):
        self.assert_stops(run_cli('--output', self.path('out.png')), 'Please specify an input image with `--input`')

    def test_input_is_not_a_file(self):
        self.assert_stops(run_cli('--input', self.path('missing.png'), '--output', self.path('out.png')),
                          'missing.png` is not a file')
        self.assert_stops(run_cli('--input', self.tmp, '--output', self.path('out.png')), 'is not a file')

    def test_input_has_an_unsupported_extension(self):
        path = self.path('image.gif')
        with open(path, 'wb') as fh:
            fh.write(b'GIF89a')
        self.assert_stops(run_cli('--input', path, '--output', self.path('out.png')),
                          'Only .jpg, .jpeg and .png images are supported')

    def test_unknown_blur_type(self):
        img_path = self.write_image('img.png', random_image())
        self.assert_stops(run_cli('--input', img_path, '--type', 'blurry', '--output', self.path('out.png')),
                          'Unknown blur type `blurry`')

    def test_depth_map_is_not_a_file(self):
        img_path = self.write_image('img.png', random_image())
        self.assert_stops(run_cli('--input', img_path, '--input_depth_map', self.path('nope.png'),
                                  '--output', self.path('out.png')),
                          'nope.png` (--input_depth_map) is not a file')

    def test_depth_map_has_an_unsupported_extension(self):
        img_path = self.write_image('img.png', random_image())
        depth_path = self.write_image('depth.bmp', gradient_depth_map())
        self.assert_stops(run_cli('--input', img_path, '--input_depth_map', depth_path,
                                  '--output', self.path('out.png')),
                          'Only .jpg, .jpeg and .png images are supported')


class TestBlur(TempDirTestCase, unittest.TestCase):
    """Without a depth map the command gives exactly what the Python functions give."""

    def setUp(self):
        super().setUp()
        self.img = random_image(size=(40, 60))
        self.img_path = self.write_image('img.png', self.img)
        self.out_path = self.path('out.png')

    def run_blur(self, *options):
        result = run_cli('--input', self.img_path, '--output', self.out_path, *options)
        self.assertEqual(result.code, 0, result.stderr)
        return result, cv2.imread(self.out_path)

    def test_motion(self):
        result, out = self.run_blur('--type', 'motion', '--motion_blur_size', '15', '--motion_blur_angle', '80')
        self.assertIn('Generating `motion` blur.', result.stdout)
        self.assertTrue(np.array_equal(out, motion_blur(self.img, size=15, angle=80)))

    def test_lens(self):
        result, out = self.run_blur('--type', 'lens', '--lens_radius', '3.5', '--lens_components', '3',
                                    '--lens_exposure_gamma', '4')
        self.assertIn('Generating `lens` blur.', result.stdout)
        self.assertTrue(np.array_equal(out, lens_blur(self.img, radius=3.5, components=3, exposure_gamma=4)))

    def test_gaussian(self):
        result, out = self.run_blur('--type', 'gaussian', '--gaussian_kernel', '8')
        self.assertIn('Generating `gaussian` blur.', result.stdout)
        self.assertTrue(np.array_equal(out, gaussian_blur(self.img, 8)))

    def test_defaults(self):
        _, out = self.run_blur()
        self.assertTrue(np.array_equal(out, motion_blur(self.img, size=100, angle=30)))
        _, out = self.run_blur('--type', 'lens')
        self.assertTrue(np.array_equal(out, lens_blur(self.img, radius=5.0, components=4, exposure_gamma=2)))
        _, out = self.run_blur('--type', 'gaussian')
        self.assertTrue(np.array_equal(out, gaussian_blur(self.img, 100, sigma=5)))

    def test_default_output_is_result_png_in_the_current_folder(self):
        cwd = os.getcwd()
        os.chdir(self.tmp)
        self.addCleanup(os.chdir, cwd)
        result = run_cli('--type', 'gaussian', '--gaussian_kernel', '5', '--input', 'img.png')
        self.assertEqual(result.code, 0)
        self.assertTrue(np.array_equal(cv2.imread(os.path.join(self.tmp, 'result.png')), gaussian_blur(self.img, 5)))

    def test_jpeg_input_and_output(self):
        jpg_path = self.write_image('photo.jpeg', self.img)
        out_path = self.path('out.jpg')
        result = run_cli('--type', 'gaussian', '--input', jpg_path, '--output', out_path)
        self.assertEqual(result.code, 0)
        self.assertEqual(cv2.imread(out_path).shape, self.img.shape)


class TestDepthBlur(TempDirTestCase, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.img = random_image()
        self.depth = gradient_depth_map()
        self.img_path = self.write_image('img.png', self.img)
        self.depth_path = self.write_image('depth.png', self.depth)
        self.out_path = self.path('out.png')

    def test_says_which_options_are_ignored(self):
        result = run_cli('--type', 'gaussian', '--input', self.img_path, '--input_depth_map', self.depth_path,
                         '--depth_max_blur', '9', '--output', self.out_path)
        self.assertEqual(result.code, 0)
        self.assertIn('Generating `gaussian` blur with depth map.', result.stdout)
        for option in ('motion_blur_size', 'lens_radius', 'gaussian_kernel'):
            self.assertIn(f'`{option}` will be ignored', result.stdout)

    def test_uses_the_angle_and_depth_options(self):
        run_cli('--type', 'motion', '--input', self.img_path, '--input_depth_map', self.depth_path,
                '--motion_blur_angle', '75', '--motion_blur_size', '3', '--depth_num_layers', '4',
                '--depth_min_blur', '2', '--depth_max_blur', '12', '--output', self.out_path)
        expected = motion_blur_with_depth_map(self.img, self.depth, angle=75, num_layers=4, min_blur=2, max_blur=12)
        self.assertTrue(np.array_equal(cv2.imread(self.out_path), expected))

    def test_jpeg_depth_map(self):
        depth_path = self.write_image('depth.jpg', self.depth)
        run_cli('--type', 'gaussian', '--input', self.img_path, '--input_depth_map', depth_path,
                '--depth_max_blur', '9', '--output', self.out_path)
        expected = gaussian_blur_with_depth_map(self.img, cv2.imread(depth_path), max_blur=9)
        self.assertTrue(np.array_equal(cv2.imread(self.out_path), expected))


if __name__ == '__main__':
    unittest.main()
