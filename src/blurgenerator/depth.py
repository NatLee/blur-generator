"""
Blur an image by depth: a depth map decides how much each pixel is blurred.

The depth map is an 8-bit image the same size as the photo, either grayscale or BGR (converted to
grayscale). Its range of values is split into ``num_layers`` layers. Every layer is blurred with its own
strength and the layers are put back together. A blur amount of 0 leaves the pixels sharp.

There are two ways to pick the strength of a layer:

- By default the blur grows with the depth value: ``min_blur`` at depth 0, ``max_blur`` at depth 255.
- With a focus (``focus=<depth>`` or ``focus_point=(x, y)``) the blur grows with the distance from the
  focus depth, in both directions, like the depth of field of a camera. Depths within ``focus_range / 2``
  of the focus are not blurred at all; just outside that band the blur is about ``min_blur``, and it
  reaches ``max_blur`` at whichever end of the 0-255 scale is farther from the band.
"""
import math
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .gaussian_blur import DEFAULT_SIGMA, gaussian_blur
from .lens_blur import DEFAULT_COMPONENTS, DEFAULT_EXPOSURE_GAMMA, lens_blur
from .motion_blur import motion_blur

#: A blur function: takes the image and a blur amount, returns the blurred image.
BlurFunction = Callable[[np.ndarray, int], np.ndarray]

#: Depth maps are 8-bit images, so depths run from 0 to DEPTH_MAX (255).
DEPTH_DTYPE = np.uint8
DEPTH_MAX = int(np.iinfo(DEPTH_DTYPE).max)

#: Defaults shared by the depth-map blur functions.
DEFAULT_NUM_LAYERS = 10
DEFAULT_MIN_BLUR = 1
DEFAULT_MAX_BLUR = 100


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


def motion_blur_with_depth_map(img: np.ndarray, depth_map: np.ndarray, angle: float = 30,
                               num_layers: int = DEFAULT_NUM_LAYERS, min_blur: int = DEFAULT_MIN_BLUR,
                               max_blur: int = DEFAULT_MAX_BLUR, *, focus: Optional[float] = None,
                               focus_range: float = 0,
                               focus_point: Optional[Sequence[int]] = None) -> np.ndarray:
    """Motion blur whose length (in pixels) follows the depth map.

    Args:
        img: Image to blur, for example from ``cv2.imread``.
        depth_map: 8-bit depth map, grayscale or BGR, the same size as ``img``.
        angle: Direction of the motion in degrees, see :func:`blurgenerator.motion_blur`.
        num_layers: Number of depth layers; more layers give smoother transitions and take longer.
        min_blur: Blur for depth 0, or right next to the focus when a focus is set.
        max_blur: Blur for depth 255, or farthest from the focus when a focus is set.
        focus: Depth (0-255) to keep sharp. The blur then grows with the distance from this depth,
            towards both 0 and 255. Off by default.
        focus_range: Width of the band of depths around the focus that stays sharp: 10 keeps
            ``focus - 5`` to ``focus + 5`` sharp. Needs ``focus`` or ``focus_point``. Default 0.
        focus_point: ``(x, y)`` pixel of the depth map whose depth becomes the focus. Use it instead of
            ``focus``.

    Returns:
        The blurred image as ``uint8``.
    """
    def blur(image: np.ndarray, amount: int) -> np.ndarray:
        return motion_blur(image, size=amount, angle=angle)

    return _blur_by_depth(img, depth_map, blur, num_layers, min_blur, max_blur, focus, focus_range, focus_point)


def lens_blur_with_depth_map(img: np.ndarray, depth_map: np.ndarray, components: int = DEFAULT_COMPONENTS,
                             exposure_gamma: float = DEFAULT_EXPOSURE_GAMMA, num_layers: int = DEFAULT_NUM_LAYERS,
                             min_blur: int = DEFAULT_MIN_BLUR, max_blur: int = DEFAULT_MAX_BLUR, *,
                             focus: Optional[float] = None, focus_range: float = 0,
                             focus_point: Optional[Sequence[int]] = None) -> np.ndarray:
    """Lens blur whose radius (in pixels) follows the depth map.

    ``components`` and ``exposure_gamma`` work as in :func:`blurgenerator.lens_blur`; the other arguments
    are described in :func:`motion_blur_with_depth_map`.
    """
    def blur(image: np.ndarray, amount: int) -> np.ndarray:
        return lens_blur(image, radius=amount, components=components, exposure_gamma=exposure_gamma)

    return _blur_by_depth(img, depth_map, blur, num_layers, min_blur, max_blur, focus, focus_range, focus_point)


def gaussian_blur_with_depth_map(img: np.ndarray, depth_map: np.ndarray, sigma: float = DEFAULT_SIGMA,
                                 num_layers: int = DEFAULT_NUM_LAYERS, min_blur: int = DEFAULT_MIN_BLUR,
                                 max_blur: int = DEFAULT_MAX_BLUR, *, focus: Optional[float] = None,
                                 focus_range: float = 0,
                                 focus_point: Optional[Sequence[int]] = None) -> np.ndarray:
    """Gaussian blur whose kernel size (in pixels) follows the depth map.

    ``sigma`` works as in :func:`blurgenerator.gaussian_blur`; the other arguments are described in
    :func:`motion_blur_with_depth_map`.
    """
    def blur(image: np.ndarray, amount: int) -> np.ndarray:
        return gaussian_blur(image, amount, sigma=sigma)

    return _blur_by_depth(img, depth_map, blur, num_layers, min_blur, max_blur, focus, focus_range, focus_point)


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
    if depth_map.dtype != DEPTH_DTYPE:
        raise ValueError(
            f'depth_map must be an 8-bit image (dtype uint8, values 0-{DEPTH_MAX}), got dtype {depth_map.dtype}.'
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


def _resolve_focus(gray: np.ndarray, focus: Optional[float], focus_range: float,
                   focus_point: Optional[Sequence[int]]) -> Optional[float]:
    """Check the focus options and return the focus depth, or ``None`` when no focus is set."""
    if focus is not None and focus_point is not None:
        raise ValueError('Give either a focus depth or a focus point, not both.')
    if focus_point is not None:
        try:
            x, y = (int(value) for value in focus_point)
        except (TypeError, ValueError):
            raise ValueError(f'The focus point must be two whole numbers (x, y), got {focus_point!r}.') from None
        height, width = gray.shape
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(
                f'The focus point ({x}, {y}) is outside the depth map, which is {width}x{height} pixels.'
            )
        focus = int(gray[y, x])
    if focus is None:
        if focus_range:
            raise ValueError('focus_range needs a focus depth or a focus point.')
        return None
    if not 0 <= focus <= DEPTH_MAX:
        raise ValueError(f'The focus depth must be between 0 and {DEPTH_MAX}, got {focus}.')
    if focus_range < 0:
        raise ValueError(f'focus_range must be 0 or more, got {focus_range}.')
    return focus


def _blur_amount(value: float, farthest: float, min_blur: int, max_blur: int) -> int:
    """Blur amount for ``value`` on a scale from 0 (``min_blur``) to ``farthest`` (``max_blur``).

    Without a focus the value is the depth and ``farthest`` is DEPTH_MAX; with a focus it is the
    distance from the sharp band and ``farthest`` the largest possible distance.
    """
    return int(map_range(value, 0, farthest, min_blur, max_blur))


def _mask(selected: np.ndarray) -> np.ndarray:
    """8-bit mask for OpenCV: 255 where ``selected`` is true, 0 elsewhere."""
    return np.where(selected, 255, 0).astype(np.uint8)


def _layer_mask(values: np.ndarray, low: int, step: int, include_low: bool) -> np.ndarray:
    """Mask of the pixels whose value lies in ``(low, low + step]``, or ``[low, low + step]``."""
    above = values >= low if include_low else values > low
    return _mask(above & (values <= low + step))


def _depth_layers(depth_map: np.ndarray, gray: np.ndarray, num_layers: int, min_blur: int,
                  max_blur: int) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(blur amount, mask)`` for every depth layer: the blur grows with the depth value."""
    step, layers = get_depth_step_and_layer(depth_map, num_layers)
    for i, value in enumerate(layers):
        yield _blur_amount(value, DEPTH_MAX, min_blur, max_blur), _layer_mask(gray, value, step, include_low=(i == 0))


def _focus_layers(gray: np.ndarray, focus: float, focus_range: float, num_layers: int, min_blur: int,
                  max_blur: int) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(blur amount, mask)`` per layer when focusing: the blur grows with the distance from the focus.

    The first layer is the sharp band (blur amount 0). The other pixels are split into layers by their
    distance from the band, so a depth just in front of the band and one just behind it get the same blur.
    """
    half_range = focus_range / 2
    # Distance from the sharp band in depth steps (0 inside the band).
    distance = np.ceil(np.maximum(np.abs(gray.astype(np.float64) - focus) - half_range, 0)).astype(np.int32)
    # max_blur is reached at whichever end of the 0-255 scale is farther from the band.
    farthest = math.ceil(max(focus - half_range, DEPTH_MAX - focus - half_range, 0))

    yield 0, _mask(distance == 0)

    outside = distance[distance > 0]
    if outside.size == 0:
        return
    step, layers = get_depth_step_and_layer(outside, num_layers)
    for i, value in enumerate(layers):
        mask = _layer_mask(distance, value, step, include_low=(i == 0))
        yield _blur_amount(value, farthest, min_blur, max_blur), mask


def _blur_by_depth(img: np.ndarray, depth_map: np.ndarray, blur: BlurFunction, num_layers: int, min_blur: int,
                   max_blur: int, focus: Optional[float] = None, focus_range: float = 0,
                   focus_point: Optional[Sequence[int]] = None) -> np.ndarray:
    """Blur ``img`` once per depth layer and keep, from each result, only the pixels of that layer."""
    _check_inputs(img, depth_map, num_layers)
    gray = _to_gray(depth_map)
    focus = _resolve_focus(gray, focus, focus_range, focus_point)
    if focus is None:
        layers = _depth_layers(depth_map, gray, num_layers, min_blur, max_blur)
    else:
        layers = _focus_layers(gray, focus, focus_range, num_layers, min_blur, max_blur)

    out = np.zeros(img.shape)
    for blur_amount, mask in layers:
        # 0 means sharp. motion_blur and lens_blur cannot do a blur of size 0 (min_blur=0 used to crash);
        # gaussian_blur of size 0 was already a plain copy.
        blurred = img if blur_amount == 0 else blur(img, blur_amount)
        layer = cv2.bitwise_and(blurred, blurred, mask=mask)
        # The result is an 8-bit image, as it has always been.
        out = cv2.add(out, layer, dtype=cv2.CV_8U)
    return out


# ----------------------------------------------------------------------------------------------------------
# Kept for compatibility. Up to 1.1.x the depth-map blurs were built from these two functions, and code
# written against those versions may import them from blurgenerator.depth. The package itself no longer
# uses them: it works with one-channel masks (_depth_layers). They can go in the next major release (2.0.0).

def get_blur_amount_and_l_mask(depth_map: np.ndarray, value: int, step: int, min_blur: int, max_blur: int,
                               include_value: bool = False) -> Tuple[int, np.ndarray]:
    """Return the blur amount and a 3-channel mask (0 or 255) for the depth layer that starts at ``value``.

    The layer covers depths in ``(value, value + step]``. The lowest layer passes ``include_value=True`` so
    it also covers ``depth == value``; otherwise pixels at the minimum depth belong to no layer.
    """
    mask = _layer_mask(_to_gray(depth_map), value, step, include_value)
    return _blur_amount(value, DEPTH_MAX, min_blur, max_blur), _three_channel_mask(mask, depth_map)


def blur_with_depth(depth_map: np.ndarray, num_layers: int = DEFAULT_NUM_LAYERS, min_blur: int = DEFAULT_MIN_BLUR,
                    max_blur: int = DEFAULT_MAX_BLUR) -> List[Tuple[int, np.ndarray]]:
    """Return ``(blur amount, 3-channel mask)`` for every depth layer of ``depth_map``."""
    layers = _depth_layers(depth_map, _to_gray(depth_map), num_layers, min_blur, max_blur)
    return [(blur_amount, _three_channel_mask(mask, depth_map)) for blur_amount, mask in layers]


def _three_channel_mask(mask: np.ndarray, depth_map: np.ndarray) -> np.ndarray:
    """``mask`` in every color channel, shaped like a BGR ``depth_map`` (a 4th channel keeps its values)."""
    if depth_map.ndim == 2 or depth_map.shape[2] == 1:
        return cv2.merge([mask, mask, mask])
    l_mask = depth_map.copy()
    l_mask[:, :, 0] = mask
    l_mask[:, :, 1] = mask
    l_mask[:, :, 2] = mask
    return l_mask
