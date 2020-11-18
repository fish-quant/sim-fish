# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate smFISH images.
"""

import numpy as np

import simfish.utils as utils

from .pattern_simulation import simulate_ground_truth
from .spot_simulation import precompute_erf
from .spot_simulation import add_spots
from .noise_simulation import add_white_noise

# TODO add foci simulation
# TODO allow image resizing to simulate subpixel localizations.


def simulate_spots(image_shape=(128, 128), image_dtype=np.uint16,
                   voxel_size_z=None, voxel_size_yx=100,
                   n=30, random_n=False,
                   sigma_z=None, sigma_yx=150, random_sigma=False,
                   amplitude=5000, random_amplitude=False, noise_level=300,
                   random_level=0.05):
    """

    Parameters
    ----------
    image_shape : Tuple[int or float] or List[int of float]
        Shape (z, y, x) or (y, x) of the image to simulate.
    image_dtype
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we
        consider a 2-d image.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    n : int
        Expected number of spots to simulate.
    random_n : bool
        Make the number of spots follow a Poisson distribution with
        expectation n, instead of a constant predefined value.
    sigma_z : int, float or None
        Standard deviation of the gaussian along the z axis, in nanometer. If
        None, we consider a 2-d image.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in nanometer.
    random_sigma : bool
        Make sigmas follow a tight normal distribution around the provided
        sigma values.
    amplitude : int or float
        Amplitude of the gaussians.
    random_amplitude : bool
        Make amplitudes follow a uniform distribution around the provided
        amplitude value.
    noise_level : int or float
        Reference level of noise background to add in the image.
    random_level : float
        Margin allowed to change gaussian parameters provided. The formula
        used is margin = parameter * random_level

    Returns
    -------
    image : np.ndarray, np.uint
        Simulated images with spots and shape (z, y, x) or (y, x).

    """
    # check parameters
    utils.check_parameter(image_shape=(tuple, list),
                          image_dtype=type,
                          voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          n=int,
                          random_n=bool,
                          sigma_z=(int, float, type(None)),
                          sigma_yx=(int, float),
                          random_sigma=bool,
                          amplitude=(int, float),
                          random_amplitude=bool,
                          noise_level=(int, float),
                          random_level=float)

    # check image dtype
    if image_dtype not in [np.uint8, np.uint16]:
        raise ValueError("'image_dtype' should be np.uint8 or np.uint16, not "
                         "{0}.".format(image_dtype))

    # check dimensions
    ndim = len(image_shape)
    if ndim not in [2, 3]:
        raise ValueError("'image_shape' should have 2 or 3 elements, not {0}."
                         .format(ndim))
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Image to simulate has 3 dimensions but "
                         "'voxel_size_z' is missing.")
    if ndim == 3 and sigma_z is None:
        raise ValueError("Image to simulate has 3 dimensions but 'sigma_z' is "
                         "missing.")

    # initialize image
    image = np.zeros(image_shape, dtype=image_dtype)

    # generate ground truth
    ground_truth = simulate_ground_truth(
        n=n, random_n=random_n, frame_shape=image_shape,
        sigma_z=sigma_z, sigma_yx=sigma_yx, random_sigma=random_sigma,
        amplitude=amplitude, random_amplitude=random_amplitude,
        random_level=random_level)

    # precompute spots if possible
    if not random_sigma:
        tables_erf = precompute_erf(
            voxel_size_z=voxel_size_z, voxel_size_yx=voxel_size_yx,
            sigma_z=sigma_z, sigma_yx=sigma_yx, grid_size=image.size)
    else:
        tables_erf = None

    # simulate spots
    image = add_spots(
        image, ground_truth,
        voxel_size_z=voxel_size_z, voxel_size_yx=voxel_size_yx,
        precomputed_gaussian=tables_erf)

    # add background noise
    image = add_white_noise(
        image, noise_level=noise_level, random_level=random_level)

    return image
