"""
Motion blur generator
"""
from random import randint
from typing import Optional

import cv2
import numpy as np


def motion_blur(img: np.ndarray, size: Optional[int] = None, angle: Optional[float] = None) -> np.ndarray:
    """Blur ``img`` as if the camera moved in a straight line while the photo was taken.

    Args:
        img: Image to blur, for example from ``cv2.imread``. Grayscale and color images both work.
        size: Length of the motion in pixels. When ``None``, a random length from 20 to 80 is used.
        angle: Direction of the motion in degrees, counter-clockwise; 0 is horizontal.
            When ``None``, a random angle from 15 to 30 is used.

    Returns:
        The blurred image, with the same shape and dtype as ``img``.
    """
    if size is None:
        size = randint(20, 80)
    if angle is None:
        angle = randint(15, 30)

    # A horizontal line through the middle of a size x size kernel, rotated by `angle`.
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[(size - 1) // 2, :] = np.ones(size, dtype=np.float32)
    center = (size / 2 - 0.5, size / 2 - 0.5)
    kernel = cv2.warpAffine(kernel, cv2.getRotationMatrix2D(center, angle, 1.0), (size, size))
    kernel = kernel * (1.0 / np.sum(kernel))

    return cv2.filter2D(img, -1, kernel)
