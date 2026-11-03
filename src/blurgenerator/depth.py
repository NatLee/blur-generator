"""
Blur an image by depth: a depth map decides how much each pixel is blurred.

The depth map is an 8-bit image the same size as the photo, either grayscale or BGR (converted to
grayscale). Its range of values is split into ``num_layers`` layers. Every layer is blurred with its own
strength, from ``min_blur`` for depth 0 to ``max_blur`` for depth 255, and the layers are put back together.
A blur amount of 0 leaves the pixels sharp.
"""
from typing import Callable, Iterator, List, Tuple

import cv2
import numpy as np

from .gaussian_blur import gaussian_blur
from .lens_blur import lens_blur
from .motion_blur import motion_blur

#: A blur function: takes the image and a blur amount, returns the blurred image.
BlurFunction = Callable[[np.ndarray, int], np.ndarray]


def map_range(value, inMin, inMax, outMin, outMax):
    """Map ``value`` linearly from the range ``[inMin, inMax]`` to ``[outMin, outMax]``."""
    return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))


def get_depth_step_and_layer(depth_map: np.ndarray, num_layers: int) -> Tuple[int, np.ndarray]:
    """Split the range of values in ``depth_map`` into about ``num_layers`` layers.

    Returns the height of a layer (``step``) and the lowest value of every layer. Layer ``i`` covers the
    values in ``(layers[i], layers[i] + step]``; the first layer also includes ``layers[0]``.
    """
    min_depth = int(np.min(depth_map))
    max_depth = int(np.max(depth_map))
    # At least 1: a depth range narrower than num_layers (or a flat depth map) gave a step of 0,
    # and range() raised "arg 3 must not be zero".
    step = max((max_depth - min_depth) // num_layers, 1)
    # A flat depth map (min == max) still gets one layer.
    layers = np.array(range(min_depth, max(max_depth, min_depth + 1), step))
    return step, layers


def get_blur_amount_and_l_mask(depth_map: np.ndarray, value: int, step: int, min_blur: int, max_blur: int,
                               include_value: bool = False) -> Tuple[int, np.ndarray]:
    """Return the blur amount and a 3-channel mask (0 or 255) for the depth layer that starts at ``value``.

    The layer covers depths in ``(value, value + step]``. The lowest layer passes ``include_value=True`` so
    it also covers ``depth == value``; otherwise pixels at the minimum depth belong to no layer.
    """
    mask = _layer_mask(_to_gray(depth_map), value, step, include_value)
    if depth_map.ndim == 2 or depth_map.shape[2] == 1:
        l_mask = cv2.merge([mask, mask, mask])
    else:
        l_mask = depth_map.copy()
        l_mask[:, :, 0] = mask
        l_mask[:, :, 1] = mask
        l_mask[:, :, 2] = mask
    return _blur_amount(value, min_blur, max_blur), l_mask


def blur_with_depth(depth_map: np.ndarray, num_layers: int = 10, min_blur: int = 1,
                    max_blur: int = 100) -> List[Tuple[int, np.ndarray]]:
    """Return ``(blur amount, 3-channel mask)`` for every depth layer of ``depth_map``."""
    step, layers = get_depth_step_and_layer(depth_map, num_layers)
    return [
        get_blur_amount_and_l_mask(depth_map, value, step, min_blur, max_blur, include_value=(i == 0))
        for i, value in enumerate(layers)
    ]


def motion_blur_with_depth_map(img: np.ndarray, depth_map: np.ndarray, angle: float = 30, num_layers: int = 10,
                               min_blur: int = 1, max_blur: int = 100) -> np.ndarray:
    """Motion blur whose length follows the depth map (``min_blur`` to ``max_blur`` pixels).

    See :func:`blurgenerator.motion_blur` for ``angle`` and the module docstring for the depth options.
    """
    def blur(image: np.ndarray, amount: int) -> np.ndarray:
        return motion_blur(image, size=amount, angle=angle)

    return _blur_by_depth(img, depth_map, blur, num_layers, min_blur, max_blur)


def lens_blur_with_depth_map(img: np.ndarray, depth_map: np.ndarray, components: int = 5,
                             exposure_gamma: float = 5, num_layers: int = 10, min_blur: int = 1,
                             max_blur: int = 100) -> np.ndarray:
    """Lens blur whose radius follows the depth map (``min_blur`` to ``max_blur`` pixels).

    See :func:`blurgenerator.lens_blur` for ``components`` and ``exposure_gamma``.
    """
    def blur(image: np.ndarray, amount: int) -> np.ndarray:
        return lens_blur(image, radius=amount, components=components, exposure_gamma=exposure_gamma)

    return _blur_by_depth(img, depth_map, blur, num_layers, min_blur, max_blur)


def gaussian_blur_with_depth_map(img: np.ndarray, depth_map: np.ndarray, sigma: float = 5, num_layers: int = 10,
                                 min_blur: int = 1, max_blur: int = 100) -> np.ndarray:
    """Gaussian blur whose kernel size follows the depth map (``min_blur`` to ``max_blur`` pixels).

    See :func:`blurgenerator.gaussian_blur` for ``sigma``.
    """
    def blur(image: np.ndarray, amount: int) -> np.ndarray:
        return gaussian_blur(image, amount, sigma=sigma)

    return _blur_by_depth(img, depth_map, blur, num_layers, min_blur, max_blur)


# ----------------------------------------------------------------------------------------------------------

def _to_gray(depth_map: np.ndarray) -> np.ndarray:
    if depth_map.ndim == 2:
        return depth_map
    if depth_map.shape[2] == 1:
        return depth_map[:, :, 0]
    return cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)


def _check_inputs(img: np.ndarray, depth_map: np.ndarray, num_layers: int) -> None:
    """Raise a clear error for inputs that would otherwise fail deep inside OpenCV."""
    for name, value in (('img', img), ('depth_map', depth_map)):
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f'{name} must be an image (a numpy array, for example from cv2.imread), '
                f'got {type(value).__name__}. cv2.imread returns None when it cannot read the file.'
            )
    if depth_map.dtype != np.uint8:
        raise ValueError(
            f'depth_map must be an 8-bit image (dtype uint8, values 0-255), got dtype {depth_map.dtype}.'
        )
    if depth_map.ndim not in (2, 3) or (depth_map.ndim == 3 and depth_map.shape[2] not in (1, 3, 4)):
        raise ValueError(f'depth_map must be a grayscale or BGR image, got an array of shape {depth_map.shape}.')
    if img.shape[:2] != depth_map.shape[:2]:
        raise ValueError(
            f'depth_map must be the same size as img: img is {img.shape[1]}x{img.shape[0]} pixels, '
            f'depth_map is {depth_map.shape[1]}x{depth_map.shape[0]}. '
            f'Resize it first, for example with cv2.resize(depth_map, ({img.shape[1]}, {img.shape[0]})).'
        )
    if num_layers < 1:
        raise ValueError(f'num_layers must be at least 1, got {num_layers}.')


def _blur_amount(value: int, min_blur: int, max_blur: int) -> int:
    """Blur amount for a depth: ``min_blur`` at depth 0, ``max_blur`` at depth 255."""
    return int(map_range(value, 0, 255, min_blur, max_blur))


def _layer_mask(values: np.ndarray, low: int, step: int, include_low: bool) -> np.ndarray:
    """Mask (0 or 255) of the pixels whose value lies in ``(low, low + step]``, or ``[low, low + step]``."""
    above = values >= low if include_low else values > low
    return np.where(above & (values <= low + step), 255, 0).astype(np.uint8)


def _depth_layers(depth_map: np.ndarray, num_layers: int, min_blur: int,
                  max_blur: int) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(blur amount, mask)`` for every depth layer: the blur grows with the depth value."""
    gray = _to_gray(depth_map)
    step, layers = get_depth_step_and_layer(depth_map, num_layers)
    for i, value in enumerate(layers):
        yield _blur_amount(value, min_blur, max_blur), _layer_mask(gray, value, step, include_low=(i == 0))


def _blur_by_depth(img: np.ndarray, depth_map: np.ndarray, blur: BlurFunction, num_layers: int, min_blur: int,
                   max_blur: int) -> np.ndarray:
    """Blur ``img`` once per depth layer and keep, from each result, only the pixels of that layer."""
    _check_inputs(img, depth_map, num_layers)
    out = np.zeros(img.shape)
    for blur_amount, mask in _depth_layers(depth_map, num_layers, min_blur, max_blur):
        # 0 means sharp. motion_blur and lens_blur cannot do a blur of size 0 (min_blur=0 used to crash);
        # gaussian_blur of size 0 was already a plain copy.
        blurred = img if blur_amount == 0 else blur(img, blur_amount)
        layer = cv2.bitwise_and(blurred, blurred, mask=mask)
        # dtype=0 (CV_8U): the result is an 8-bit image, as it has always been.
        out = cv2.add(out, layer, dtype=0)
    return out
