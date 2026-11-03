"""Regression tests for bugs fixed after 1.1.0."""
import contextlib
import io
import re
import unittest

import cv2
import numpy as np

from blurgenerator import (
    motion_blur,
    lens_blur,
    gaussian_blur,
    motion_blur_with_depth_map,
    lens_blur_with_depth_map,
    gaussian_blur_with_depth_map,
)
from blurgenerator.cli import main as cli_main

from .helpers import TempDirTestCase, gradient_depth_map, random_image, run_cli


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

    def test_min_blur_zero_keeps_the_lowest_layer_sharp(self):
        # 1.1.0 crashed with min_blur=0 for motion (IndexError) and lens (ZeroDivisionError) blur.
        img = random_image()
        depth = np.zeros((40, 40, 3), dtype=np.uint8)
        depth[:, 20:] = 255
        for fn in DEPTH_FUNCTIONS:
            with self.subTest(fn=fn.__name__):
                out = fn(img, depth, num_layers=2, min_blur=0, max_blur=5)
                self.assertTrue(np.array_equal(out[:, :20], img[:, :20]))
                self.assertFalse(np.array_equal(out[:, 20:], img[:, 20:]))


class TestDepthMapInputs(unittest.TestCase):

    def setUp(self):
        self.img = random_image()
        self.depth = gradient_depth_map()

    def test_grayscale_depth_map_gives_the_same_result_as_bgr(self):
        # 1.1.0 only accepted 3-channel depth maps; cv2.imread(path, cv2.IMREAD_GRAYSCALE) failed.
        gray = cv2.cvtColor(self.depth, cv2.COLOR_BGR2GRAY)
        for fn in DEPTH_FUNCTIONS:
            with self.subTest(fn=fn.__name__):
                expected = fn(self.img, self.depth, num_layers=3, min_blur=1, max_blur=9)
                for depth in (gray, gray[:, :, np.newaxis]):
                    self.assertTrue(np.array_equal(fn(self.img, depth, num_layers=3, min_blur=1, max_blur=9),
                                                   expected))

    def test_depth_map_of_another_size_is_a_clear_error(self):
        depth = cv2.resize(self.depth, (30, 20))
        for fn in DEPTH_FUNCTIONS:
            with self.subTest(fn=fn.__name__):
                with self.assertRaisesRegex(ValueError, r'img is 40x40 pixels, depth_map is 30x20'):
                    fn(self.img, depth)

    def test_depth_map_must_be_an_image(self):
        with self.assertRaisesRegex(ValueError, r'grayscale or BGR image, got an array of shape \(40, 40, 2\)'):
            gaussian_blur_with_depth_map(self.img, self.depth[:, :, :2])

    def test_depth_map_must_be_8_bit(self):
        with self.assertRaisesRegex(ValueError, r'8-bit .* got dtype uint16'):
            gaussian_blur_with_depth_map(self.img, self.depth.astype(np.uint16))

    def test_missing_depth_map_is_a_clear_error(self):
        # What you get when cv2.imread cannot find the depth map file.
        with self.assertRaisesRegex(TypeError, r'depth_map must be an image .* got NoneType'):
            gaussian_blur_with_depth_map(self.img, None)

    def test_num_layers_must_be_at_least_one(self):
        with self.assertRaisesRegex(ValueError, r'num_layers must be at least 1, got 0'):
            gaussian_blur_with_depth_map(self.img, self.depth, num_layers=0)


class TestBlurArguments(unittest.TestCase):

    def test_motion_blur_size_below_one_is_a_clear_error(self):
        # 1.1.0 raised IndexError for size=0.
        with self.assertRaisesRegex(ValueError, r'size must be at least 1 pixel, got 0'):
            motion_blur(random_image(), size=0, angle=0)

    def test_lens_blur_radius_zero_is_a_clear_error(self):
        # 1.1.0 raised ZeroDivisionError for radius=0.
        with self.assertRaisesRegex(ValueError, r'radius must be greater than 0, got 0'):
            lens_blur(random_image(), radius=0)

    def test_lens_blur_accepts_grayscale_images(self):
        # 1.1.0 raised "axes don't match array" for 2D images; motion and gaussian blur already took them.
        gray = random_image()[:, :, 0]
        out = lens_blur(gray, radius=3)
        self.assertEqual(out.shape, gray.shape)
        expected = lens_blur(cv2.merge([gray, gray, gray]), radius=3)[:, :, 0]
        self.assertTrue(np.array_equal(out, expected))


class TestCliDepthMap(TempDirTestCase, unittest.TestCase):

    def test_cli_depth_blur_matches_python_api(self):
        img = random_image()
        depth = gradient_depth_map()
        cases = [
            ('motion', motion_blur_with_depth_map, dict(angle=30)),
            ('lens', lens_blur_with_depth_map, dict(components=4, exposure_gamma=2)),
            ('gaussian', gaussian_blur_with_depth_map, dict(sigma=5)),
        ]
        img_path = self.write_image('img.png', img)
        depth_path = self.write_image('depth.png', depth)
        for blur_type, fn, kwargs in cases:
            with self.subTest(blur_type=blur_type):
                out_path = self.path(blur_type + '.png')
                run_cli('--type', blur_type, '--input', img_path, '--input_depth_map', depth_path,
                        '--depth_num_layers', '3', '--depth_min_blur', '1', '--depth_max_blur', '9',
                        '--output', out_path)
                expected = fn(img, depth, num_layers=3, min_blur=1, max_blur=9, **kwargs)
                self.assertTrue(np.array_equal(cv2.imread(out_path), expected))

    def test_help_shows_the_real_default_of_depth_num_layers(self):
        # 1.1.0's --help said "Default is 3", but the CLI has always used 10 layers.
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit):
                cli_main(['--help'])
        help_text = re.sub(r'\x1b\[[0-9;]*m', '', out.getvalue())  # drop colors (Python 3.14+)
        match = re.search(r'--depth_num_layers DEPTH_NUM_LAYERS\s+[^-]*?Default\s+is\s+(\d+)\.', help_text)
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), '10')

        # ... and 10 is what the CLI really uses when --depth_num_layers is not given.
        img = random_image()
        depth = gradient_depth_map()
        img_path = self.write_image('img.png', img)
        depth_path = self.write_image('depth.png', depth)
        out_path = self.path('out.png')
        run_cli('--type', 'gaussian', '--input', img_path, '--input_depth_map', depth_path,
                '--depth_max_blur', '9', '--output', out_path)
        expected = gaussian_blur_with_depth_map(img, depth, num_layers=10, min_blur=1, max_blur=9)
        self.assertTrue(np.array_equal(cv2.imread(out_path), expected))


class TestCliErrors(TempDirTestCase, unittest.TestCase):

    def test_upper_case_extensions_are_accepted(self):
        # 1.1.0 rejected photo.JPG / photo.PNG with "Only support common types of image".
        img = random_image()
        img_path = self.write_image('photo.PNG', img)
        out_path = self.path('out.png')
        result = run_cli('--type', 'gaussian', '--gaussian_kernel', '5', '--input', img_path, '--output', out_path)
        self.assertEqual(result.code, 0)
        self.assertTrue(np.array_equal(cv2.imread(out_path), gaussian_blur(img, 5)))

    def test_unreadable_image_is_a_clear_error(self):
        # 1.1.0 crashed inside OpenCV with a traceback when the file was not an image.
        bad = self.path('broken.png')
        with open(bad, 'w') as fh:
            fh.write('not an image')
        result = run_cli('--input', bad, '--output', self.path('out.png'))
        self.assertEqual(result.code, 1)
        self.assertIn('Could not read', result.stderr)

        good = self.write_image('img.png', random_image())
        result = run_cli('--input', good, '--input_depth_map', bad, '--output', self.path('out.png'))
        self.assertEqual(result.code, 1)
        self.assertIn('(--input_depth_map) as an image', result.stderr)

    def test_output_in_a_missing_folder_is_reported(self):
        # 1.1.0 wrote nothing and said nothing.
        img_path = self.write_image('img.png', random_image())
        result = run_cli('--type', 'gaussian', '--input', img_path, '--output', self.path('nope/out.png'))
        self.assertEqual(result.code, 1)
        self.assertIn('Could not write', result.stderr)

    def test_output_with_an_unknown_extension_is_a_clear_error(self):
        # 1.1.0 crashed inside OpenCV with a traceback.
        img_path = self.write_image('img.png', random_image())
        result = run_cli('--type', 'gaussian', '--input', img_path, '--output', self.path('out.txt'))
        self.assertEqual(result.code, 1)
        self.assertIn('image extension', result.stderr)

    def test_invalid_blur_setting_is_a_clear_error(self):
        # 1.1.0 crashed with IndexError for --motion_blur_size 0.
        img_path = self.write_image('img.png', random_image())
        result = run_cli('--input', img_path, '--motion_blur_size', '0', '--output', self.path('out.png'))
        self.assertEqual(result.code, 1)
        self.assertIn('size must be at least 1 pixel', result.stderr)

    def test_depth_map_of_another_size_is_a_clear_error(self):
        img_path = self.write_image('img.png', random_image())
        depth_path = self.write_image('depth.png', cv2.resize(gradient_depth_map(), (30, 20)))
        result = run_cli('--input', img_path, '--input_depth_map', depth_path, '--output', self.path('out.png'))
        self.assertEqual(result.code, 1)
        self.assertIn('depth_map must be the same size as img', result.stderr)


if __name__ == '__main__':
    unittest.main()
