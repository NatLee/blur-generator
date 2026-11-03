"""Regression tests for bugs fixed after 1.1.0."""
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from blurgenerator import (
    lens_blur,
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


if __name__ == '__main__':
    unittest.main()
