"""Regression tests for bugs fixed after 1.1.0."""
import unittest

import numpy as np

from blurgenerator import lens_blur


def random_image(seed=0, size=(40, 40)):
    return np.random.RandomState(seed).randint(1, 255, size=size + (3,), dtype=np.uint8)


class TestLensBlurComponents(unittest.TestCase):

    def test_more_than_six_components_uses_six(self):
        # 1.1.0 raised IndexError for components > 6 instead of clamping to the 6-component preset.
        img = random_image()
        expected = lens_blur(img, radius=3, components=6)
        for components in (7, 10):
            result = lens_blur(img, radius=3, components=components)
            self.assertTrue(np.array_equal(result, expected))


if __name__ == '__main__':
    unittest.main()
