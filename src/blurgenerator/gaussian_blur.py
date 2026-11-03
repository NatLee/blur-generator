"""
Gaussian blur generator
"""
import cv2
import numpy as np


def gaussian_blur(img: np.ndarray, kernel: int, sigma: float = 5) -> np.ndarray:
    """Blur ``img`` with a Gaussian filter.

    Args:
        img: Image to blur, for example from ``cv2.imread``. Grayscale and color images both work.
        kernel: Width and height of the filter window in pixels. OpenCV needs an odd size, so an even
            value is rounded up (100 behaves like 101).
        sigma: Standard deviation of the Gaussian in pixels; it decides how strong the blur is.
            The window only cuts the Gaussian off, so a kernel much wider than ``6 * sigma`` barely
            changes the result.
            With ``sigma=0`` OpenCV derives sigma from the kernel size instead.

    Returns:
        The blurred image, with the same shape and dtype as ``img``.
    """
    if kernel % 2 == 0:
        kernel += 1
    return cv2.GaussianBlur(img, (kernel, kernel), sigma)
