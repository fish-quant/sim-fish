# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for simfish.spot_simulation module.
"""

import pytest

import numpy as np
import simfish as sim


# TODO add test for warnings and errors raised

@pytest.mark.parametrize("dtype", [
    np.uint8, np.uint16, np.float32, np.float64])
def test_spot_simulation(dtype):
    # 2D simulation
    ground_truth = np.array([[5., 5., 100., 200.]])
    image = np.zeros((11, 11), dtype=dtype)
    new_image = sim.add_spots(
        image,
        ground_truth=ground_truth,
        voxel_size=(100, 100))
    assert new_image.dtype == dtype
    assert new_image.shape == (11, 11)
    mask_spot = np.zeros_like(new_image).astype(bool)
    mask_spot[3:8, 3:8] = True
    assert np.all(new_image[mask_spot] >= 0)
    assert new_image[~mask_spot].sum() == 0

    # 3D simulation
    ground_truth = np.array([[5., 5., 5., 100, 100., 200.]])
    image = np.zeros((11, 11, 11), dtype=dtype)
    new_image = sim.add_spots(
        image,
        ground_truth=ground_truth,
        voxel_size=(100, 100, 100))
    assert new_image.dtype == dtype
    assert new_image.shape == (11, 11, 11)
    mask_spot = np.zeros_like(new_image).astype(bool)
    mask_spot[3:8, 3:8, 3:8] = True
    assert np.all(new_image[mask_spot] >= 0)
    assert new_image[~mask_spot].sum() == 0
