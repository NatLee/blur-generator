"""Regression tests for bugs fixed after 1.1.0."""
import contextlib
import io
import os
import re
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from blurgenerator import (
    lens_blur,
    gaussian_blur,
    motion_blur_with_depth_map,
    lens_blur_with_depth_map,
    gaussian_blur_with_depth_map,
)
from blurgenerator.cli import main as cli_main


def random_image(seed=0, size=(40, 40)):
    return np.random.RandomState(seed).randint(1, 255, size=size + (3,), dtype=np.uint8)


def gradient_depth_map(size=(40, 40)):
    """3-channel depth map that goes from 0 on the left to 255 on the right."""
    row = np.linspace(0, 255, size[1]).astype(np.uint8)
    gray = np.tile(row, (size[0], 1))
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def run_cli(*args):
    """Run the blurgenerator command in-process and return what it printed."""
    out = io.StringIO()
    with mock.patch.object(sys, 'argv', ['blurgenerator'] + list(args)), contextlib.redirect_stdout(out):
        cli_main()
    return out.getvalue()


class TestLensBlurComponents(unittest.TestCase):

    def test_more_than_six_components_uses_six(self):
        # 1.1.0 raised IndexError for components > 6 instead of clamping to the 6-component preset.
        img = random_image()
        expected = lens_blur(img, radius=3, components=6)
        for components in (7, 10):
            result = lens_blur(img, radius=3, components=components)
            self.assertTrue(np.array_equal(result, expected))


DEPTH_FUNCTIONS = (motion_blur_with_depth_map, lens_blur_with_depth_map, gaussian_blur_with_depth_map)


def count_black_pixels(img):
    return int((img.reshape(-1, img.shape[-1]).sum(axis=1) == 0).sum())


class TestDepthMap(unittest.TestCase):

    def setUp(self):
        # Plain gray image: any black pixel in the output is a pixel the depth blur dropped.
        self.img = np.full((40, 40, 3), 120, dtype=np.uint8)

    def test_pixels_at_the_lowest_depth_are_not_black(self):
        # 1.1.0 left every pixel whose depth equals the minimum of the map black.
        depth = np.zeros((40, 40, 3), dtype=np.uint8)
        depth[:, 20:] = 200
        for fn in DEPTH_FUNCTIONS:
            with self.subTest(fn=fn.__name__):
                out = fn(self.img, depth, num_layers=3, min_blur=1, max_blur=9)
                self.assertEqual(count_black_pixels(out), 0)

    def test_depth_range_narrower_than_num_layers(self):
        # 1.1.0 raised "range() arg 3 must not be zero" when max - min < num_layers.
        depth = np.full((40, 40, 3), 100, dtype=np.uint8)
        depth[:, 20:] = 105
        for fn in DEPTH_FUNCTIONS:
            with self.subTest(fn=fn.__name__):
                out = fn(self.img, depth, num_layers=10, min_blur=1, max_blur=9)
                self.assertEqual(count_black_pixels(out), 0)

    def test_flat_depth_map_blurs_the_whole_image_once(self):
        # A depth map with a single value is one layer: the same as blurring the whole image
        # with the blur amount for that depth (128 -> 5 for min_blur=1, max_blur=9).
        img = random_image()
        depth = np.full((40, 40, 3), 128, dtype=np.uint8)
        out = gaussian_blur_with_depth_map(img, depth, sigma=5, num_layers=3, min_blur=1, max_blur=9)
        self.assertTrue(np.array_equal(out, gaussian_blur(img, 5, sigma=5)))


class TestCliDepthMap(unittest.TestCase):

    def test_cli_depth_blur_matches_python_api(self):
        img = random_image()
        depth = gradient_depth_map()
        cases = [
            ('motion', motion_blur_with_depth_map, dict(angle=30)),
            ('lens', lens_blur_with_depth_map, dict(components=4, exposure_gamma=2)),
            ('gaussian', gaussian_blur_with_depth_map, dict(sigma=5)),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, 'img.png')
            depth_path = os.path.join(tmp, 'depth.png')
            cv2.imwrite(img_path, img)
            cv2.imwrite(depth_path, depth)
            for blur_type, fn, kwargs in cases:
                with self.subTest(blur_type=blur_type):
                    out_path = os.path.join(tmp, blur_type + '.png')
                    run_cli('--type', blur_type, '--input', img_path, '--input_depth_map', depth_path,
                            '--depth_num_layers', '3', '--depth_min_blur', '1', '--depth_max_blur', '9',
                            '--output', out_path)
                    expected = fn(img, depth, num_layers=3, min_blur=1, max_blur=9, **kwargs)
                    self.assertTrue(np.array_equal(cv2.imread(out_path), expected))

    def test_help_shows_the_real_default_of_depth_num_layers(self):
        # 1.1.0's --help said "Default is 3", but the CLI has always used 10 layers.
        out = io.StringIO()
        with mock.patch.object(sys, 'argv', ['blurgenerator', '--help']), contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                cli_main()
        help_text = re.sub(r'\x1b\[[0-9;]*m', '', out.getvalue())  # drop colors (Python 3.14+)
        match = re.search(r'Layer\s+for\s+depth\s+blur\.\s+Default\s+is\s+(\d+)\.', help_text)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), '10')

        # ... and 10 is what the CLI really uses when --depth_num_layers is not given.
        img = random_image()
        depth = gradient_depth_map()
        with tempfile.TemporaryDirectory() as tmp:
            img_path = os.path.join(tmp, 'img.png')
            depth_path = os.path.join(tmp, 'depth.png')
            out_path = os.path.join(tmp, 'out.png')
            cv2.imwrite(img_path, img)
            cv2.imwrite(depth_path, depth)
            run_cli('--type', 'gaussian', '--input', img_path, '--input_depth_map', depth_path,
                    '--depth_max_blur', '9', '--output', out_path)
            expected = gaussian_blur_with_depth_map(img, depth, num_layers=10, min_blur=1, max_blur=9)
            self.assertTrue(np.array_equal(cv2.imread(out_path), expected))


if __name__ == '__main__':
    unittest.main()
