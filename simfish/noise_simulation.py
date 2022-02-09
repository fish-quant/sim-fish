# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate background noise.
"""

import numpy as np
import bigfish.stack as stack


# TODO add illumination bias

def add_white_noise(image, noise_level, random_noise=0.05):
    """Generate and add white noise to an image.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    noise_level : int or float
        Reference level of noise background to add in the image.
    random_noise : int or float, default=0.05
        Background noise follows a normal distribution around the provided
        noise values. The scale used is:

        .. math::
                \\mbox{scale} = \\mbox{noise_level} * \\mbox{random_noise}

    Returns
    -------
    noised_image : np.ndarray
        Noised image with shape (z, y, x) or (y, x).

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_parameter(
        noise_level=(int, float),
        random_noise=(int, float))

    # original dtype
    original_dtype = image.dtype
    if np.issubdtype(original_dtype, np.integer):
        original_max_bound = np.iinfo(original_dtype).max
    else:
        original_max_bound = np.finfo(original_dtype).max

    # compute scale
    scale = noise_level * random_noise

    # generate noise
    noise_raw = np.random.normal(loc=noise_level, scale=scale, size=image.size)
    noise_raw[noise_raw < 0] = 0.
    noise_raw = np.random.poisson(lam=noise_raw, size=noise_raw.size)
    noise = np.reshape(noise_raw, image.shape)

    # add noise
    noised_image = image.astype(np.float64) + noise
    noised_image = np.clip(noised_image, 0, original_max_bound)
    noised_image = noised_image.astype(original_dtype)

    return noised_image
