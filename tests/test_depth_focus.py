"""Depth of field for the depth-map blur: keep one depth sharp, blur in both directions (issue #5)."""
import itertools
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

from .helpers import TempDirTestCase, gradient_depth_map, random_image, run_cli


def reference_depth_blur(img, depth_map, blur, num_layers, min_blur, max_blur):
    """The depth-map blur as it was before the focus options (main at c0e1561), frozen for comparison."""
    dm = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
    min_depth = int(np.min(depth_map))
    max_depth = int(np.max(depth_map))
    step = max((max_depth - min_depth) // num_layers, 1)
    layers = np.array(range(min_depth, max(max_depth, min_depth + 1), step))
    out = np.zeros(img.shape)
    for i, value in enumerate(layers):
        m = np.zeros(dm.shape)
        if i == 0:
            m[dm >= value] = 255
        else:
            m[dm > value] = 255
        m[dm > (value + step)] = 0
        l_mask = depth_map.copy()
        l_mask[:, :, 0] = m[:, :]
        l_mask[:, :, 1] = m[:, :]
        l_mask[:, :, 2] = m[:, :]
        blur_amount = int(min_blur + (((value - 0) / (255 - 0)) * (max_blur - min_blur)))
        blurred = blur(img, blur_amount)
        _, mask = cv2.threshold(l_mask, 100, 255, cv2.THRESH_BINARY)
        layer = cv2.bitwise_and(blurred, blurred, mask=mask[:, :, 0])
        out = cv2.add(out, layer, dtype=0)
    return out


def bgr(gray):
    return cv2.cvtColor(np.ascontiguousarray(gray, dtype=np.uint8), cv2.COLOR_GRAY2BGR)


def depth_per_column(height=8):
    """Depth map whose column x has depth x (0-255), so a column is a depth."""
    return bgr(np.tile(np.arange(256), (height, 1)))


def bands(*depths, width=20, height=20):
    """Depth map made of vertical bands, one per depth value."""
    return bgr(np.hstack([np.full((height, width), depth) for depth in depths]))


BLUR_TYPES = (
    ('motion', motion_blur_with_depth_map, dict(angle=30), lambda img, n: motion_blur(img, size=n, angle=30)),
    ('lens', lens_blur_with_depth_map, dict(components=4, exposure_gamma=2),
     lambda img, n: lens_blur(img, radius=n, components=4, exposure_gamma=2)),
    ('gaussian', gaussian_blur_with_depth_map, dict(sigma=5), lambda img, n: gaussian_blur(img, n, sigma=5)),
)


class TestWithoutFocusNothingChanges(unittest.TestCase):
    """With the focus options unset, the output must be bit-for-bit what it was before they existed."""

    def test_output_is_bit_for_bit_the_same_as_before(self):
        img = random_image(size=(30, 45))
        noise = np.random.RandomState(7).randint(0, 256, (30, 45), dtype=np.uint8)
        depth_maps = {
            'gradient': gradient_depth_map(size=(30, 45)),
            'two values': bgr(np.where(np.arange(45) < 20, 0, 200)[np.newaxis].repeat(30, axis=0)),
            'narrow range': bgr(np.where(np.arange(45) < 20, 100, 105)[np.newaxis].repeat(30, axis=0)),
            'flat': bgr(np.full((30, 45), 128)),
            'noise': bgr(noise),
        }
        settings = [(3, 1, 9), (10, 1, 30), (5, 20, 2)]
        for (name, fn, kwargs, blur), (map_name, depth), (layers, lo, hi) in itertools.product(
                BLUR_TYPES, depth_maps.items(), settings):
            with self.subTest(blur=name, depth_map=map_name, layers=layers, min_blur=lo, max_blur=hi):
                expected = reference_depth_blur(img, depth, blur, layers, lo, hi)
                actual = fn(img, depth, num_layers=layers, min_blur=lo, max_blur=hi, **kwargs)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertTrue(np.array_equal(actual, expected))
                explicit = fn(img, depth, num_layers=layers, min_blur=lo, max_blur=hi,
                              focus=None, focus_range=0, focus_point=None, **kwargs)
                self.assertTrue(np.array_equal(explicit, expected))

    def test_defaults_are_unchanged(self):
        img = random_image()
        # Depths 0-21 only, so the blur stays small and the test fast; the defaults still apply.
        depth = gradient_depth_map() // 12
        default_blurs = {
            motion_blur_with_depth_map: lambda i, n: motion_blur(i, size=n, angle=30),
            lens_blur_with_depth_map: lambda i, n: lens_blur(i, radius=n, components=5, exposure_gamma=5),
            gaussian_blur_with_depth_map: lambda i, n: gaussian_blur(i, n, sigma=5),
        }
        for fn, blur in default_blurs.items():
            with self.subTest(fn=fn.__name__):
                expected = reference_depth_blur(img, depth, blur, num_layers=10, min_blur=1, max_blur=100)
                self.assertTrue(np.array_equal(fn(img, depth), expected))


class TestFocus(unittest.TestCase):

    def test_pixels_at_the_focus_depth_stay_sharp(self):
        img = random_image(size=(8, 256))
        depth = depth_per_column()
        for name, fn, kwargs, _ in BLUR_TYPES:
            with self.subTest(blur=name):
                out = fn(img, depth, num_layers=5, min_blur=3, max_blur=9, focus=77, **kwargs)
                self.assertTrue(np.array_equal(out[:, 77], img[:, 77]))
                self.assertFalse(np.array_equal(out[:, 76], img[:, 76]))
                self.assertFalse(np.array_equal(out[:, 78], img[:, 78]))

    def test_focus_range_keeps_a_band_of_depths_sharp(self):
        # The example from issue #5: 45 to 55 fully in focus.
        img = random_image(size=(8, 256))
        depth = depth_per_column()
        out = gaussian_blur_with_depth_map(img, depth, num_layers=5, min_blur=3, max_blur=9,
                                           focus=50, focus_range=10)
        self.assertTrue(np.array_equal(out[:, 45:56], img[:, 45:56]))
        for column in (44, 56, 0, 255):
            self.assertFalse(np.array_equal(out[:, column], img[:, column]), column)

    def test_blur_grows_the_same_way_in_front_of_and_behind_the_focus(self):
        img = random_image(size=(20, 60))
        depth = bands(60, 100, 140)
        out = gaussian_blur_with_depth_map(img, depth, num_layers=10, min_blur=1, max_blur=31, focus=100)
        # Both outer bands are 40 away from the focus. The farthest end of the scale is 255 (155 away),
        # so both get 1 + 40 / 155 * (31 - 1) = 8.7 -> 8.
        blurred = gaussian_blur(img, 8)
        self.assertTrue(np.array_equal(out[:, 20:40], img[:, 20:40]))
        self.assertTrue(np.array_equal(out[:, :20], blurred[:, :20]))
        self.assertTrue(np.array_equal(out[:, 40:], blurred[:, 40:]))

        # Without a focus the same depth map blurs the near band more than the far one.
        out = gaussian_blur_with_depth_map(img, depth, num_layers=10, min_blur=1, max_blur=31)
        self.assertTrue(np.array_equal(out[:, :20], gaussian_blur(img, 8)[:, :20]))  # 1 + 60/255*30 = 8.1
        self.assertTrue(np.array_equal(out[:, 40:], gaussian_blur(img, 17)[:, 40:]))  # 1 + 140/255*30 = 17.5

    def test_the_far_end_of_the_scale_gets_max_blur(self):
        img = random_image(size=(20, 40))
        depth = bands(50, 255)
        out = gaussian_blur_with_depth_map(img, depth, min_blur=1, max_blur=21, focus=50)
        self.assertTrue(np.array_equal(out[:, :20], img[:, :20]))
        self.assertTrue(np.array_equal(out[:, 20:], gaussian_blur(img, 21)[:, 20:]))

    def test_focus_point_uses_the_depth_under_that_pixel(self):
        img = random_image(size=(8, 256))
        depth = depth_per_column()
        for name, fn, kwargs, _ in BLUR_TYPES:
            with self.subTest(blur=name):
                expected = fn(img, depth, num_layers=5, min_blur=1, max_blur=9, focus=77, focus_range=6, **kwargs)
                out = fn(img, depth, num_layers=5, min_blur=1, max_blur=9, focus_point=(77, 3), focus_range=6,
                         **kwargs)
                self.assertTrue(np.array_equal(out, expected))

    def test_a_band_that_covers_every_depth_leaves_the_image_sharp(self):
        img = random_image()
        depth = gradient_depth_map()
        for name, fn, kwargs, _ in BLUR_TYPES:
            with self.subTest(blur=name):
                out = fn(img, depth, focus=128, focus_range=256, **kwargs)
                self.assertTrue(np.array_equal(out, img))

    def test_grayscale_depth_map_and_min_blur_zero(self):
        img = random_image(size=(8, 256))
        gray = depth_per_column()[:, :, 0]
        for name, fn, kwargs, _ in BLUR_TYPES:
            with self.subTest(blur=name):
                out = fn(img, gray, num_layers=4, min_blur=0, max_blur=6, focus=128, **kwargs)
                self.assertTrue(np.array_equal(out[:, 120:137], img[:, 120:137]))
                self.assertFalse(np.array_equal(out[:, :5], img[:, :5]))
                self.assertFalse(np.array_equal(out[:, -5:], img[:, -5:]))

    def test_invalid_focus_options_are_clear_errors(self):
        img = random_image()
        depth = gradient_depth_map()
        cases = [
            (dict(focus=256), r'between 0 and 255, got 256'),
            (dict(focus=-1), r'between 0 and 255, got -1'),
            (dict(focus=50, focus_range=-2), r'focus_range must be 0 or more, got -2'),
            (dict(focus=50, focus_point=(1, 1)), r'either a focus depth or a focus point, not both'),
            (dict(focus_point=(40, 0)), r'\(40, 0\) is outside the depth map, which is 40x40 pixels'),
            (dict(focus_point=(0, -1)), r'outside the depth map'),
            (dict(focus_point=(1,)), r'two whole numbers'),
            (dict(focus_range=10), r'focus_range needs a focus depth or a focus point'),
        ]
        for kwargs, message in cases:
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                with self.assertRaisesRegex(ValueError, message):
                    gaussian_blur_with_depth_map(img, depth, **kwargs)

    def test_focus_options_are_keyword_only(self):
        # So that a future positional argument can never be mistaken for a focus depth.
        with self.assertRaises(TypeError):
            gaussian_blur_with_depth_map(random_image(), gradient_depth_map(), 5, 10, 1, 100, 55)


class TestCliFocus(TempDirTestCase, unittest.TestCase):

    def setUp(self):
        super().setUp()
        self.img = random_image(size=(8, 256))
        self.depth = depth_per_column()
        self.img_path = self.write_image('img.png', self.img)
        self.depth_path = self.write_image('depth.png', self.depth)

    def run_depth(self, blur_type, *options):
        out_path = self.path('out.png')
        result = run_cli('--type', blur_type, '--input', self.img_path, '--input_depth_map', self.depth_path,
                         '--depth_num_layers', '4', '--depth_min_blur', '2', '--depth_max_blur', '8',
                         '--output', out_path, *options)
        return result, cv2.imread(out_path)

    def test_cli_focus_matches_python_api(self):
        cli_kwargs = dict(motion=dict(angle=30), lens=dict(components=4, exposure_gamma=2), gaussian=dict(sigma=5))
        for name, fn, _, _ in BLUR_TYPES:
            with self.subTest(blur=name):
                expected = fn(self.img, self.depth, num_layers=4, min_blur=2, max_blur=8, focus=100,
                              focus_range=20, **cli_kwargs[name])
                result, out = self.run_depth(name, '--depth_focus', '100', '--depth_focus_range', '20')
                self.assertEqual(result.code, 0)
                self.assertTrue(np.array_equal(out, expected))
                result, out = self.run_depth(name, '--depth_focus_point', '100', '5', '--depth_focus_range', '20')
                self.assertEqual(result.code, 0)
                self.assertTrue(np.array_equal(out, expected))

    def test_focus_options_without_a_depth_map_are_ignored(self):
        out_path = self.path('out.png')
        result = run_cli('--type', 'gaussian', '--gaussian_kernel', '5', '--input', self.img_path,
                         '--depth_focus', '50', '--output', out_path)
        self.assertEqual(result.code, 0)
        self.assertIn('will be ignored without `--input_depth_map`', result.stdout)
        self.assertTrue(np.array_equal(cv2.imread(out_path), gaussian_blur(self.img, 5)))

    def test_invalid_focus_is_a_clear_error(self):
        cases = [
            (('--depth_focus', '300'), 'between 0 and 255'),
            (('--depth_focus_point', '999', '0'), 'outside the depth map'),
            (('--depth_focus', '10', '--depth_focus_point', '1', '1'), 'not both'),
            (('--depth_focus_range', '10'), 'needs a focus depth or a focus point'),
        ]
        for options, message in cases:
            with self.subTest(options=options):
                result, _ = self.run_depth('gaussian', *options)
                self.assertEqual(result.code, 1)
                self.assertIn(message, result.stderr)


if __name__ == '__main__':
    unittest.main()
