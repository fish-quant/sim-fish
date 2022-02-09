# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for simfish.noise_simulation module.
"""

import pytest

import numpy as np
import simfish as sim

from numpy.testing import assert_array_equal


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("ndim", [2, 3])
def test_add_white_noise(dtype, ndim):
    # simulate base image
    if ndim == 2:
        image = np.zeros((5, 5), dtype=dtype)
    else:
        image = np.zeros((5, 5, 5), dtype=dtype)

    # add random noise
    noised_image = sim.add_white_noise(
        image, noise_level=0, random_noise=0.05)
    assert noised_image.dtype == image.dtype
    assert noised_image.shape == image.shape

    # add nothing
    noised_image = sim.add_white_noise(
        image, noise_level=0, random_noise=0)
    assert noised_image.dtype == image.dtype
    assert noised_image.shape == image.shape
    assert_array_equal(noised_image, image)

    # add constant noise
    noised_image = sim.add_white_noise(
        image, noise_level=5, random_noise=0)
    assert noised_image.dtype == image.dtype
    assert noised_image.shape == image.shape
