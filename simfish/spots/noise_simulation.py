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
    image : np.ndarray, np.uint
        Image with shape (z, y, x) or (y, x).
    noise_level : int or float
        Reference level of noise background to add in the image.
    random_noise : int or float
        Background noise follows a normal distribution around the provided
        noise values. The scale used is scale = noise_level * random_noise

    Returns
    -------
    noised_image : np.ndarray, np.uint
        Noised image with shape (z, y, x) or (y, x).

    """
    # check parameters
    stack.check_array(image, ndim=[2, 3], dtype=[np.uint8, np.uint16])
    stack.check_parameter(
        noise_level=(int, float),
        random_noise=(int, float))

    # compute scale
    scale = noise_level * random_noise

    # generate noise
    noise = np.random.normal(loc=noise_level, scale=scale, size=image.size)
    noise = np.reshape(noise, image.shape)

    # add noise
    noised_image = image.astype(np.float64) + noise
    noised_image = np.clip(noised_image, 0, np.iinfo(image.dtype).max)
    noised_image = noised_image.astype(image.dtype)

    return noised_image
