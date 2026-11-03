"""
Lens blur generator

Approximates the round "bokeh" blur of a camera lens with a sum of separable complex Gaussian kernels,
following <http://yehar.com/blog/?p=1495> (circularly symmetric convolution and lens blur).
"""
import math
import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from typing import Dict, List, Tuple

import cv2
import numpy as np

# These scales bring the size of the components below to roughly the requested radius (hand-tuned).
kernel_scales = [1.4, 1.2, 1.2, 1.2, 1.2, 1.2]

# Kernel parameters a, b, A, B for 1 to 6 components, from <http://yehar.com/blog/?p=1495>.
kernel_params = [
    # 1 component
    [[0.862325, 1.624835, 0.767583, 1.862321]],

    # 2 components
    [[0.886528, 5.268909, 0.411259, -0.548794],
     [1.960518, 1.558213, 0.513282, 4.56111]],

    # 3 components
    [[2.17649, 5.043495, 1.621035, -2.105439],
     [1.019306, 9.027613, -0.28086, -0.162882],
     [2.81511, 1.597273, -0.366471, 10.300301]],

    # 4 components
    [[4.338459, 1.553635, -5.767909, 46.164397],
     [3.839993, 4.693183, 9.795391, -15.227561],
     [2.791880, 8.178137, -3.048324, 0.302959],
     [1.342190, 12.328289, 0.010001, 0.244650]],

    # 5 components
    [[4.892608, 1.685979, -22.356787, 85.91246],
     [4.71187, 4.998496, 35.918936, -28.875618],
     [4.052795, 8.244168, -13.212253, -1.578428],
     [2.929212, 11.900859, 0.507991, 1.816328],
     [1.512961, 16.116382, 0.138051, -0.01]],

    # 6 components
    [[5.143778, 2.079813, -82.326596, 111.231024],
     [5.612426, 6.153387, 113.878661, 58.004879],
     [5.982921, 9.802895, 39.479083, -162.028887],
     [6.505167, 11.059237, -71.286026, 95.027069],
     [3.869579, 14.81052, 1.405746, -3.704914],
     [2.201904, 19.032909, -0.152784, -0.107988]],
]


def get_parameters(component_count: int = 2) -> Tuple[List[Dict[str, float]], float]:
    """Return the kernel parameters (``a``, ``b``, ``A``, ``B`` per component) and scale for a component count.

    Counts outside 1-6 use the nearest preset: values above 6 use the 6-component kernel.
    """
    parameter_index = max(0, min(component_count - 1, len(kernel_params) - 1))
    parameter_dictionaries = [dict(zip(['a', 'b', 'A', 'B'], b)) for b in kernel_params[parameter_index]]
    return parameter_dictionaries, kernel_scales[parameter_index]


def complex_kernel_1d(radius: float, scale: float, a: float, b: float) -> np.ndarray:
    """Build a 1D complex kernel of the given radius; ``scale`` corrects the radius, ``a`` and ``b`` shape it."""
    kernel_radius = int(math.ceil(radius))
    kernel_size = kernel_radius * 2 + 1
    ax = np.linspace(-radius, radius, kernel_size, dtype=np.float32)
    ax = ax * scale * (1 / radius)
    kernel_complex = np.zeros((kernel_size,), dtype=np.complex64)
    kernel_complex.real = np.exp(-a * (ax**2)) * np.cos(b * (ax**2))
    kernel_complex.imag = np.exp(-a * (ax**2)) * np.sin(b * (ax**2))
    return kernel_complex.reshape((1, kernel_size))


def normalise_kernels(kernels: List[np.ndarray], params: List[Dict[str, float]]) -> np.ndarray:
    """Scale the kernels so that the combined 2D kernel (``A * real + B * imag``) sums to 1."""
    total = 0

    for k, p in zip(kernels, params):
        for i in range(k.shape[1]):
            for j in range(k.shape[1]):
                total += p['A'] * (k[0, i].real * k[0, j].real - k[0, i].imag * k[0, j].imag) + \
                         p['B'] * (k[0, i].real * k[0, j].imag + k[0, i].imag * k[0, j].real)

    scalar = 1 / math.sqrt(total)
    kernels = np.asarray(kernels) * scalar

    return kernels


def weighted_sum(kernel: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """Combine the real and imaginary parts of a filtered image, weighted by ``A`` and ``B``."""
    return np.add(kernel.real * params['A'], kernel.imag * params['B'])


def multiply_kernel(kernel: np.ndarray) -> np.ndarray:
    """Turn a 1D kernel into the equivalent 2D kernel (slower than the separable filter; for inspection)."""
    kernel_size = kernel.shape[1]
    a = np.repeat(kernel, kernel_size, 0)
    b = np.repeat(kernel.transpose(), kernel_size, 1)
    return np.multiply(a, b)


def filter_task(idx: int, channel: int, img_channel: np.ndarray, component: np.ndarray,
                component_params: Dict[str, float]) -> Tuple[int, int, np.ndarray]:
    """Filter one image channel with one complex kernel component (rows, then columns).

    For two complex signals f and g, f * g = Re(f)*Re(g) - Im(f)*Im(g) + i [Re(f)*Im(g) + Im(f)*Re(g)].
    Returns ``(idx, channel, weighted_sum)`` so callers can tell the results apart.
    """
    component_real = np.real(component)
    component_imag = np.imag(component)

    component_real_t = component_real.transpose()
    component_imag_t = component_imag.transpose()

    inter_real = cv2.filter2D(img_channel, -1, component_real, borderType=cv2.BORDER_REPLICATE)
    inter_imag = cv2.filter2D(img_channel, -1, component_imag, borderType=cv2.BORDER_REPLICATE)

    final_1 = cv2.filter2D(inter_real, -1, component_real_t, borderType=cv2.BORDER_REPLICATE)
    final_2 = cv2.filter2D(inter_real, -1, component_imag_t, borderType=cv2.BORDER_REPLICATE)
    final_3 = cv2.filter2D(inter_imag, -1, component_real_t, borderType=cv2.BORDER_REPLICATE)
    final_4 = cv2.filter2D(inter_imag, -1, component_imag_t, borderType=cv2.BORDER_REPLICATE)

    final = final_1 - final_4 + 1j * (final_2 + final_3)
    return idx, channel, weighted_sum(final, component_params)


def lens_blur(img: np.ndarray, radius: float = 3.0, components: int = 5, exposure_gamma: float = 5.0) -> np.ndarray:
    """Blur ``img`` like an out-of-focus camera lens: bright spots turn into round discs ("bokeh").

    Args:
        img: 8-bit image to blur, for example from ``cv2.imread``. Grayscale and color images both work.
        radius: Radius of the blur disc in pixels; decimals are allowed.
        components: Number of kernel components, 1 to 6. More components give a rounder, cleaner disc
            and take longer. Values above 6 use 6.
        exposure_gamma: Brightens highlights before blurring and undoes it afterwards, so bright spots
            bloom into visible discs. 1 turns this off.

    Returns:
        The blurred image as ``uint8``, with the same shape as ``img``.

    Raises:
        ValueError: if ``radius`` is not above 0.
    """
    if radius <= 0:
        raise ValueError(f'radius must be greater than 0, got {radius}.')
    grayscale = img.ndim == 2
    if grayscale:
        img = img[:, :, np.newaxis]

    img = img / 255.
    # One contiguous float32 plane per channel.
    img = np.ascontiguousarray(img.transpose(2, 0, 1), dtype=np.float32)

    parameters, scale = get_parameters(component_count=components)
    kernels = [complex_kernel_1d(radius, scale, params['a'], params['b']) for params in parameters]
    # The combination of all kernels applied in 2D must sum to 1.
    kernels = normalise_kernels(kernels, parameters)

    # Increase exposure to highlight bright spots.
    img = np.power(img, exposure_gamma)

    # Filter every channel with every component in parallel, then stack the channels back in order and
    # add the components up in order (the order matters for the exact floating point result).
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [
            [
                executor.submit(filter_task, idx, channel, img[channel], kernel, params)
                for channel in range(img.shape[0])
            ]
            for idx, (kernel, params) in enumerate(zip(kernels, parameters))
        ]
        component_output = [
            np.stack([future.result()[2] for future in channel_futures])
            for channel_futures in futures
        ]
    output_image = reduce(np.add, component_output)

    # Reverse the exposure. Clip small negatives first: the complex kernels are not perfect.
    output_image = np.clip(output_image, 0, None)
    output_image = np.power(output_image, 1.0 / exposure_gamma)
    output_image = np.clip(output_image, 0, 1)
    output_image *= 255
    output_image = output_image.transpose(1, 2, 0).astype(np.uint8)
    return output_image[:, :, 0] if grayscale else output_image
