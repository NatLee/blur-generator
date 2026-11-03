"""
Motion blur generator
"""
from random import randint
from typing import Optional

import cv2
import numpy as np

#: Ranges (inclusive) that the length and the angle are drawn from when the caller leaves them out.
RANDOM_SIZE = (20, 80)
RANDOM_ANGLE = (15, 30)


def motion_blur(img: np.ndarray, size: Optional[int] = None, angle: Optional[float] = None) -> np.ndarray:
    """Blur ``img`` as if the camera moved in a straight line while the photo was taken.

    Args:
        img: Image to blur, for example from ``cv2.imread``. Grayscale and color images both work.
        size: Length of the motion in pixels. When ``None``, a random length from 20 to 80 is used.
        angle: Direction of the motion in degrees, counter-clockwise; 0 is horizontal.
            When ``None``, a random angle from 15 to 30 is used.

    Returns:
        The blurred image, with the same shape and dtype as ``img``.

    Raises:
        ValueError: if ``size`` is below 1.
    """
    if size is None:
        size = randint(*RANDOM_SIZE)
    if angle is None:
        angle = randint(*RANDOM_ANGLE)
    if size < 1:
        raise ValueError(f'size must be at least 1 pixel, got {size}.')

    # A horizontal line through the middle of a size x size kernel, rotated by `angle`.
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    center = (size / 2 - 0.5, size / 2 - 0.5)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D(center, angle, 1.0), (size, size))
    kernel = kernel * (1.0 / np.sum(kernel))

    return cv2.filter2D(img, -1, kernel)
