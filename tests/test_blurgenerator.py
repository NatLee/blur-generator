"""Behaviour of the three blur types on their own (no depth map)."""
import random
import unittest

import cv2
import numpy as np

from blurgenerator import motion_blur, lens_blur, gaussian_blur
from blurgenerator.lens_blur import complex_kernel_1d, get_parameters, multiply_kernel, normalise_kernels

from .helpers import random_image


def sharpness(img):
    """Variance of the Laplacian: lower means blurrier."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def dot(size=41, half_width=0):
    """Black image with a white dot (or small square) in the middle."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    c = size // 2
    img[c - half_width:c + half_width + 1, c - half_width:c + half_width + 1] = 255
    return img


class TestBlurGenerator(unittest.TestCase):

    def test_motion_blur(self):
        rgb = np.random.randint(255, size=(50, 50, 3), dtype=np.uint8)
        blur_img = motion_blur(rgb)
        self.assertFalse(np.array_equal(rgb, blur_img))

    def test_lens_blur(self):
        rgb = np.random.randint(255, size=(50, 50, 3), dtype=np.uint8)
        blur_img = lens_blur(rgb)
        self.assertFalse(np.array_equal(rgb, blur_img))

    def test_gaussian_blur(self):
        rgb = np.random.randint(255, size=(50, 50, 3), dtype=np.uint8)
        blur_img = gaussian_blur(rgb, kernel=3)
        self.assertFalse(np.array_equal(rgb, blur_img))


class TestEveryBlurType(unittest.TestCase):

    BLURS = {
        'motion': lambda img: motion_blur(img, size=15, angle=30),
        'lens': lambda img: lens_blur(img, radius=6),
        'gaussian': lambda img: gaussian_blur(img, 15),
    }

    def test_shape_and_dtype_are_kept(self):
        for name, blur in self.BLURS.items():
            for img in (random_image(size=(30, 50)), random_image(size=(30, 50))[:, :, 0]):
                with self.subTest(blur=name, shape=img.shape):
                    out = blur(img)
                    self.assertEqual(out.shape, img.shape)
                    self.assertEqual(out.dtype, np.uint8)

    def test_a_flat_image_stays_flat(self):
        # Every kernel sums to 1, so blurring a single color must not change it.
        flat = np.full((40, 40, 3), 128, dtype=np.uint8)
        for name, blur in self.BLURS.items():
            with self.subTest(blur=name):
                self.assertTrue(np.array_equal(blur(flat), flat))

    def test_the_input_is_not_modified(self):
        img = random_image()
        original = img.copy()
        for name, blur in self.BLURS.items():
            with self.subTest(blur=name):
                blur(img)
                self.assertTrue(np.array_equal(img, original))


class TestMotionBlur(unittest.TestCase):

    def test_spreads_along_the_angle(self):
        # angle 0 smears a dot into a horizontal line of `size` pixels, angle 90 into a vertical one.
        horizontal = motion_blur(dot(), size=9, angle=0)[:, :, 0]
        rows, cols = np.nonzero(horizontal)
        self.assertEqual((rows.min(), rows.max(), cols.min(), cols.max()), (20, 20, 16, 24))

        vertical = motion_blur(dot(), size=9, angle=90)[:, :, 0]
        rows, cols = np.nonzero(vertical)
        self.assertEqual((rows.min(), rows.max(), cols.min(), cols.max()), (16, 24, 20, 20))

        diagonal = motion_blur(dot(), size=9, angle=45)[:, :, 0]
        rows, cols = np.nonzero(diagonal)
        self.assertGreater(rows.max() - rows.min(), 4)
        self.assertGreater(cols.max() - cols.min(), 4)

    def test_longer_motion_blurs_more(self):
        img = random_image(size=(60, 60))
        values = [sharpness(motion_blur(img, size=size, angle=30)) for size in (3, 9, 27)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_size_one_changes_nothing(self):
        img = random_image()
        self.assertTrue(np.array_equal(motion_blur(img, size=1, angle=30), img))

    def test_size_and_angle_are_random_when_not_given(self):
        img = random_image()
        random.seed(3)
        size, angle = random.randint(20, 80), random.randint(15, 30)
        random.seed(3)
        self.assertTrue(np.array_equal(motion_blur(img), motion_blur(img, size=size, angle=angle)))


class TestLensBlur(unittest.TestCase):

    def test_a_bright_point_becomes_a_round_disc(self):
        img = dot(size=81, half_width=1)
        yy, xx = np.mgrid[:81, :81]
        distance = np.hypot(yy - 40, xx - 40)
        for radius in (5, 10, 15):
            with self.subTest(radius=radius):
                lit = lens_blur(img, radius=radius, components=5, exposure_gamma=3)[:, :, 0] > 0
                self.assertTrue(lit[distance <= radius - 2].all())
                self.assertFalse(lit[distance > 1.5 * radius + 3].any())

    def test_bigger_radius_blurs_more(self):
        img = random_image(size=(60, 60))
        values = [sharpness(lens_blur(img, radius=radius)) for radius in (1, 2, 5, 10)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_every_component_count_works(self):
        img = random_image()
        results = [lens_blur(img, radius=4, components=components) for components in range(1, 7)]
        for components, result in enumerate(results, 1):
            with self.subTest(components=components):
                self.assertEqual(result.shape, img.shape)
                self.assertFalse(np.array_equal(result, img))
        # Each preset is a different kernel.
        self.assertFalse(np.array_equal(results[0], results[5]))

    def test_decimal_radius_is_accepted(self):
        img = random_image()
        self.assertEqual(lens_blur(img, radius=2.5).shape, img.shape)

    def test_multiply_kernel_builds_the_2d_kernel(self):
        parameters, scale = get_parameters(2)
        kernel = complex_kernel_1d(4, scale, parameters[0]['a'], parameters[0]['b'])
        self.assertEqual(kernel.shape, (1, 9))
        self.assertTrue(np.allclose(multiply_kernel(kernel), np.outer(kernel[0], kernel[0])))

    def test_exposure_gamma_makes_highlights_bloom(self):
        img = dot(size=61, half_width=1)
        plain = lens_blur(img, radius=10, exposure_gamma=1)
        bloom = lens_blur(img, radius=10, exposure_gamma=5)
        self.assertGreater(int((bloom > 0).sum()), int((plain > 0).sum()))


def lens_kernel(radius, components):
    """The 2D kernel lens_blur filters with: sum over the components of A * real + B * imaginary part."""
    parameters, scale = get_parameters(components)
    kernels = normalise_kernels(
        [complex_kernel_1d(radius, scale, params['a'], params['b']) for params in parameters], parameters)
    return sum(params['A'] * np.outer(k[0], k[0]).real + params['B'] * np.outer(k[0], k[0]).imag
               for k, params in zip(kernels, parameters))


def disc_area(radius, components):
    """Size of the blur disc: the pixels the kernel lights to at least 10% of its peak.

    The kernel can ring a little (small negative and positive lobes), so a moment such as the RMS radius
    can go down while the disc grows; counting the bright pixels cannot.
    """
    kernel = lens_kernel(radius, components)
    return int((kernel >= 0.1 * kernel.max()).sum())


def complex_kernel_1d_up_to_1_2(radius, scale, a, b):
    """complex_kernel_1d as it was up to 1.2.0, frozen for comparison: it sampled the kernel at
    2 * ceil(radius) + 1 points spread evenly over [-radius, radius], so 2.5 gave the kernel of 3."""
    kernel_radius = int(np.ceil(radius))
    kernel_size = kernel_radius * 2 + 1
    ax = np.linspace(-radius, radius, kernel_size, dtype=np.float32)
    ax = ax * scale * (1 / radius)
    kernel = np.zeros((kernel_size,), dtype=np.complex64)
    kernel.real = np.exp(-a * (ax**2)) * np.cos(b * (ax**2))
    kernel.imag = np.exp(-a * (ax**2)) * np.sin(b * (ax**2))
    return kernel.reshape((1, kernel_size))


class TestLensRadius(unittest.TestCase):
    """Decimal radii really blur by that much (up to 1.2.0 they acted like the next whole number)."""

    def test_a_decimal_radius_lies_between_the_whole_radii_around_it(self):
        for components in range(1, 7):
            for low in (2, 4):
                with self.subTest(components=components, radius=low + 0.5):
                    self.assertLess(disc_area(low, components), disc_area(low + 0.5, components))
                    self.assertLess(disc_area(low + 0.5, components), disc_area(low + 1, components))

    def test_the_disc_never_shrinks_as_the_radius_grows(self):
        radii = np.arange(1, 12.01, 0.125)
        for components in range(1, 7):
            with self.subTest(components=components):
                areas = [disc_area(radius, components) for radius in radii]
                self.assertTrue(all(np.diff(areas) >= 0), dict(zip(radii.tolist(), areas)))
                self.assertGreater(areas[-1], areas[0])

    def test_a_decimal_radius_changes_the_image(self):
        # Up to 1.2.0, radius=2.5 gave exactly the image of radius=3.
        img = random_image()
        results = [lens_blur(img, radius=radius) for radius in (2, 2.5, 3)]
        self.assertFalse(np.array_equal(results[1], results[0]))
        self.assertFalse(np.array_equal(results[1], results[2]))

    def test_whole_radii_give_the_same_kernel_as_before(self):
        for components in range(1, 7):
            parameters, scale = get_parameters(components)
            for radius in (1, 2, 3, 5, 12, 40):
                for params in parameters:
                    with self.subTest(components=components, radius=radius):
                        old = complex_kernel_1d_up_to_1_2(radius, scale, params['a'], params['b'])
                        new = complex_kernel_1d(radius, scale, params['a'], params['b'])
                        self.assertEqual(new.dtype, old.dtype)
                        self.assertTrue(np.array_equal(new, old))


class TestGaussianBlur(unittest.TestCase):

    def test_is_opencv_gaussian_blur(self):
        img = random_image()
        self.assertTrue(np.array_equal(gaussian_blur(img, 7, sigma=2), cv2.GaussianBlur(img, (7, 7), 2)))
        self.assertTrue(np.array_equal(gaussian_blur(img, 7), cv2.GaussianBlur(img, (7, 7), 5)))

    def test_even_kernel_is_rounded_up(self):
        img = random_image()
        self.assertTrue(np.array_equal(gaussian_blur(img, 4), gaussian_blur(img, 5)))
        self.assertTrue(np.array_equal(gaussian_blur(img, 100), gaussian_blur(img, 101)))

    def test_bigger_kernel_blurs_more_with_sigma_zero(self):
        # sigma=0 lets OpenCV derive sigma from the kernel size.
        img = random_image(size=(60, 60))
        values = [sharpness(gaussian_blur(img, kernel, sigma=0)) for kernel in (3, 9, 27)]
        self.assertEqual(values, sorted(values, reverse=True))

    def test_kernel_zero_or_one_changes_nothing(self):
        img = random_image()
        self.assertTrue(np.array_equal(gaussian_blur(img, 0), img))
        self.assertTrue(np.array_equal(gaussian_blur(img, 1), img))


if __name__ == '__main__':
    unittest.main()
