# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for simfish.image_simulation module.
"""

import pytest

import numpy as np
import simfish as sim


# TODO add test for downscale_image
# TODO add test for errors

@pytest.mark.parametrize("image_dtype", [
    np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("subpixel_factors_value", [
    None, 3])
def test_simulate_images(image_dtype, subpixel_factors_value):
    # 2D image
    if subpixel_factors_value is not None:
        subpixel_factors = (subpixel_factors_value,) * 2
    else:
        subpixel_factors = None
    generator = sim.simulate_images(
        n_images=5,
        ndim=2,
        n_spots=10,
        image_shape=(10, 10),
        image_dtype=image_dtype,
        subpixel_factors=subpixel_factors)
    n = 0
    for image, ground_truth in generator:
        assert image.shape == (10, 10)
        assert image.dtype == image_dtype
        assert len(ground_truth) == 10
        n += 1
    assert n == 5

    # 3D image
    if subpixel_factors_value is not None:
        subpixel_factors = (subpixel_factors_value,) * 3
    else:
        subpixel_factors = None
    generator = sim.simulate_images(
        n_images=5,
        ndim=3,
        n_spots=10,
        image_shape=(10, 10, 10),
        image_dtype=image_dtype,
        subpixel_factors=subpixel_factors,
        voxel_size=(100, 100, 100),
        sigma=(150, 150, 150))
    n = 0
    for image, ground_truth in generator:
        assert image.shape == (10, 10, 10)
        assert image.dtype == image_dtype
        assert len(ground_truth) == 10
        n += 1
    assert n == 5



@pytest.mark.parametrize("image_dtype", [
    np.uint8, np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("subpixel_factors_value", [
    None, 3])
def test_simulate_image(image_dtype, subpixel_factors_value):
    # 2D image
    if subpixel_factors_value is not None:
        subpixel_factors = (subpixel_factors_value,) * 2
    else:
        subpixel_factors = None
    image, ground_truth = sim.simulate_image(
        ndim=2,
        n_spots=10,
        image_shape=(10, 10),
        image_dtype=image_dtype,
        subpixel_factors=subpixel_factors)
    assert image.shape == (10, 10)
    assert image.dtype == image_dtype
    assert len(ground_truth) == 10

    # 3D image
    if subpixel_factors_value is not None:
        subpixel_factors = (subpixel_factors_value,) * 3
    else:
        subpixel_factors = None
    image, ground_truth = sim.simulate_image(
        ndim=3,
        n_spots=10,
        image_shape=(10, 10, 10),
        image_dtype=image_dtype,
        subpixel_factors=subpixel_factors,
        voxel_size=(100, 100, 100),
        sigma=(150, 150, 150))
    assert image.shape == (10, 10, 10)
    assert image.dtype == image_dtype
    assert len(ground_truth) == 10
