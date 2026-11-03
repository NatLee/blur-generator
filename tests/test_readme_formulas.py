"""Check the formulas in the README ("The maths") against what the package really does.

Each formula is written out again here with NumPy, straight from the README, and compared with the
package's output. Motion and Gaussian blur are linear filters, so blurring an image that is 1 in one
pixel and 0 elsewhere (an impulse) gives back their kernel, which is compared with the formula.
"""
import math
import unittest

import cv2
import numpy as np

from blurgenerator import gaussian_blur, gaussian_blur_with_depth_map, lens_blur, motion_blur
from blurgenerator.lens_blur import kernel_params, kernel_scales

from .helpers import random_image


def kernel_from_impulse(blur, size):
    """Blur a float impulse and read the kernel back (OpenCV filters by correlation, anchored at size // 2)."""
    p = 2 * size
    impulse = np.zeros((4 * size, 4 * size), dtype=np.float32)
    impulse[p, p] = 1
    out = blur(impulse)
    a = size // 2
    return out[p + a - size + 1:p + a + 1, p + a - size + 1:p + a + 1][::-1, ::-1].astype(np.float64)


# --- Motion blur ---------------------------------------------------------------------------------------------

def motion_kernel(n, angle):
    """README: M = R_theta L / sum(R_theta L), L = a line of ones on row floor((n - 1) / 2)."""
    c = (n - 1) / 2
    row = (n - 1) // 2
    theta = math.radians(angle)
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    u = c + (x - c) * math.cos(theta) - (y - c) * math.sin(theta)
    v = c + (x - c) * math.sin(theta) + (y - c) * math.cos(theta)
    # L between pixels, by bilinear interpolation; 0 outside the n x n kernel.
    line = np.clip(1 - np.abs(v - row), 0, None) * np.clip(np.minimum(u + 1, n - u), 0, 1)
    return line / line.sum()


# --- Gaussian blur -------------------------------------------------------------------------------------------

def gaussian_kernel(k, sigma):
    """README: k' = k rounded up to odd, h = (k' - 1) / 2, G(i, j) = g(i) g(j)."""
    k = k + 1 if k % 2 == 0 else k
    h = (k - 1) // 2
    t = np.arange(-h, h + 1)
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return np.outer(g, g)


# --- Lens blur -----------------------------------------------------------------------------------------------

def lens_kernel(radius, components):
    """README: u = s x / r for x = -ceil(r)..ceil(r); K = sum of A Re[k(x) k(y)] + B Im[k(x) k(y)], sum 1."""
    preset = max(0, min(components - 1, len(kernel_params) - 1))
    s = kernel_scales[preset]
    x = np.arange(-math.ceil(radius), math.ceil(radius) + 1)
    u = s * x / radius
    kernel = 0
    for a, b, big_a, big_b in kernel_params[preset]:
        k = np.exp(-a * u ** 2) * (np.cos(b * u ** 2) + 1j * np.sin(b * u ** 2))
        kk = np.outer(k, k)
        kernel = kernel + big_a * kk.real + big_b * kk.imag
    return kernel / kernel.sum()


def lens_formula(img, radius, components, gamma):
    """README: O = 255 min(1, max(0, (I / 255)^gamma * K)^(1 / gamma)), edges repeated, rounded down."""
    kernel = lens_kernel(radius, components)
    linear = (img.astype(np.float64) / 255) ** gamma
    filtered = cv2.filter2D(linear, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    return (np.minimum(1, np.maximum(0, filtered) ** (1 / gamma)) * 255).astype(np.uint8)


# --- Blur by depth -------------------------------------------------------------------------------------------

def depth_formula(img, depth, num_layers, min_blur, max_blur, focus=None, focus_range=0):
    """README "Blur by depth" and "Focus": which blur amount each pixel gets, then pick it from that blur."""
    d = depth.astype(np.float64)
    if focus is None:
        values, top = d, 255
        sharp = np.zeros(d.shape, dtype=bool)
    else:
        half = focus_range / 2
        values = np.ceil(np.maximum(np.abs(d - focus) - half, 0))
        top = math.ceil(max(focus - half, 255 - focus - half, 0))
        sharp = values == 0
    out = img.copy()  # the sharp band keeps its pixels
    if sharp.all():
        return out
    low, high = values[~sharp].min(), values[~sharp].max()
    step = max((high - low) // num_layers, 1)
    starts = np.arange(low, max(high, low + 1), step)
    for i, start in enumerate(starts):
        layer = ~sharp & (values <= start + step) & ((values > start) | ((values == start) & (i == 0)))
        amount = math.floor(min_blur + start / top * (max_blur - min_blur))
        blurred = img if amount == 0 else gaussian_blur(img, amount)
        out[layer] = blurred[layer]
    return out


class TestReadmeFormulas(unittest.TestCase):

    def test_motion_blur_kernel(self):
        for n, angle in ((9, 0), (9, 30), (12, 45), (15, 90), (20, -15), (31, 77)):
            with self.subTest(size=n, angle=angle):
                code = kernel_from_impulse(lambda img: motion_blur(img, size=n, angle=angle), n)
                want = motion_kernel(n, angle)
                self.assertAlmostEqual(code.sum(), 1, places=5)
                # Only 32-bit float rounding apart (measured: 3.5e-6 of the largest value, sizes 3-50, 8 angles).
                self.assertLessEqual(np.abs(code - want).max(), 1e-5 * want.max())

    def test_gaussian_blur_kernel(self):
        for k, sigma in ((7, 2), (8, 2), (31, 5), (101, 5), (15, 1.5)):
            with self.subTest(kernel=k, sigma=sigma):
                want = gaussian_kernel(k, sigma)
                code = kernel_from_impulse(lambda img: gaussian_blur(img, k, sigma=sigma), want.shape[0])
                # Measured: 2.2e-9 apart at most.
                self.assertTrue(np.allclose(code, want, rtol=0, atol=1e-7))

    def test_lens_blur(self):
        img = random_image(size=(32, 40))
        for radius, components, gamma in ((2, 1, 1), (2.5, 3, 2), (4, 5, 5), (3.25, 6, 3), (6, 8, 5)):
            with self.subTest(radius=radius, components=components, gamma=gamma):
                code = lens_blur(img, radius=radius, components=components, exposure_gamma=gamma).astype(int)
                want = lens_formula(img, radius, components, gamma).astype(int)
                # The package filters in 32-bit floats, one component and one direction at a time; the formula
                # in 64-bit floats in one go. Rounding down then differs by 1 for a few pixels (measured: at most
                # 0.08% of them, radius 1.5-9.5, 1-8 components, gamma 1-5).
                self.assertLessEqual(np.abs(code - want).max(), 1)
                self.assertLess(np.mean(code != want), 0.01)

    def test_blur_by_depth(self):
        img = random_image(size=(20, 64))
        gray = np.tile(np.linspace(0, 255, 64).astype(np.uint8), (20, 1))
        depth = cv2.merge([gray] * 3)
        cases = [
            dict(num_layers=3, min_blur=1, max_blur=9),
            dict(num_layers=10, min_blur=0, max_blur=15),
            dict(num_layers=5, min_blur=20, max_blur=2),
            dict(num_layers=4, min_blur=2, max_blur=12, focus=100, focus_range=20),
            dict(num_layers=6, min_blur=0, max_blur=9, focus=0),
            dict(num_layers=3, min_blur=1, max_blur=9, focus=128, focus_range=256),
        ]
        for case in cases:
            with self.subTest(**case):
                code = gaussian_blur_with_depth_map(img, depth, **case)
                self.assertTrue(np.array_equal(code, depth_formula(img, gray, **case)))


if __name__ == '__main__':
    unittest.main()
