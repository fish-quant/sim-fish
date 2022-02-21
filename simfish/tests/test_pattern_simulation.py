# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Unitary tests for simfish.pattern_simulation module.
"""

import numpy as np
import simfish as sim


# TODO add test for `probability_map` parameter
# TODO add test for warnings and errors raised
# TODO add test for `build_probability_map` function
# TODO add test for `simulate_localization_pattern` function

def test_format():
    # 2D simulation
    ground_truth = sim.simulate_ground_truth(
        ndim=2,
        n_spots=8,
        frame_shape=(32, 32),
        voxel_size=(100, 100),
        sigma=(150, 150))
    assert ground_truth.dtype == np.float64
    assert ground_truth.shape == (8, 4)

    # 3D simulation
    ground_truth = sim.simulate_ground_truth(
        ndim=3,
        n_spots=12,
        frame_shape=(10, 32, 32),
        voxel_size=(300, 100, 100),
        sigma=(150, 150, 150))
    assert ground_truth.dtype == np.float64
    assert ground_truth.shape == (12, 6)


def test_randomness():
    # randomness
    _ = sim.simulate_ground_truth(
        ndim=2,
        n_spots=30,
        random_n_spots=True,
        random_n_clusters=True,
        random_n_spots_cluster=True,
        frame_shape=(32, 32),
        voxel_size=(100, 100),
        sigma=(150, 150))
    assert True

    # no randomness
    ground_truth = sim.simulate_ground_truth(
        ndim=2,
        n_spots=30,
        random_n_spots=False,
        random_n_clusters=False,
        random_n_spots_cluster=False,
        frame_shape=(32, 32),
        voxel_size=(100, 100),
        sigma=(150, 150),
        random_sigma=0,
        amplitude=5000,
        random_amplitude=0)
    assert len(ground_truth) == 30
    assert np.all(ground_truth[:, 2] == 150)
    assert np.all(ground_truth[:, 3] == 5000)
