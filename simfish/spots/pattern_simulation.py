# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate spots patterns.
"""

import numpy as np

import simfish.utils as utils


def simulate_ground_truth(n=30, random_n=False, frame_shape=(128, 128),
                          sigma_z=None, sigma_yx=150, random_sigma=False,
                          amplitude=5000, random_amplitude=False,
                          random_level=0.05):
    """ Simulate ground truth information about the simulated spots like their
    coordinates, standard deviations and amplitude.

    Parameters
    ----------
    n : int
        Expected number of spots to simulate.
    random_n : bool
        Make the number of spots follow a Poisson distribution with
        expectation n, instead of a constant  predefined value.
    frame_shape : Tuple[int or float] or List[int of float]
        Shape (z, y, x) or (y, x) of the image to simulate.
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
    random_level : float
        Margin allowed to change gaussian parameters provided. The formula
        used is margin = parameter * random_level

    Returns
    -------
    ground_truth : np.ndarray
        Ground truth array with shape (nb_spots, 4) or (nb_spots, 6).
        - coordinate_z (optional)
        - coordinate_y
        - coordinate_x
        - sigma_z (optional)
        - sigma_yx
        - amplitude

    """
    # check parameters
    utils.check_parameter(n=int,
                          random_n=bool,
                          frame_shape=(tuple, list),
                          sigma_z=(int, float, type(None)),
                          sigma_yx=(int, float),
                          random_sigma=bool,
                          amplitude=(int, float),
                          random_amplitude=bool,
                          random_level=float)

    # check dimensions
    ndim = len(frame_shape)
    if ndim not in [2, 3]:
        raise ValueError("'frame_shape' should have 2 or 3 elements, not {0}."
                         .format(ndim))
    if ndim == 3 and sigma_z is None:
        raise ValueError("Frame to simulate has 3 dimensions but 'sigma_z' is "
                         "missing.")

    # generate number of spots to simulate
    nb_spots = _get_nb_spots(n, random_n)

    # simulate positions
    positions_z, positions_y, positions_x = _get_spots_coordinates(
        frame_shape, ndim, nb_spots)

    # generate sigma values
    sigmas_z, sigmas_yx = _get_sigma(ndim, sigma_z, sigma_yx, random_sigma,
                                     random_level, nb_spots)

    # generate amplitude values
    amplitudes = _get_amplitude(amplitude, random_amplitude,
                                random_level, nb_spots)

    # stack and format ground truth
    if ndim == 3:
        ground_truth = np.stack((positions_z, positions_y, positions_x,
                                 sigmas_z, sigmas_yx, amplitudes)).T
    else:
        ground_truth = np.stack((positions_y, positions_x,
                                 sigmas_yx, amplitudes)).T

    return ground_truth


def _get_nb_spots(n, random_n):
    """Generate  number of spots to simulate.

    Parameters
    ----------
    n : int
        Expected number of spots to simulate.
    random_n : bool
        Make the number of spots follow a Poisson distribution with
        expectation n, instead of a constant  predefined value.

    Returns
    -------
    nb_spots : int
        Number of spots to simulate.

    """
    # generate number of spots to simulate
    if random_n:
        nb_spots = int(np.random.poisson(lam=n, size=1))
    else:
        nb_spots = n

    return nb_spots


def _get_spots_coordinates(frame_shape, ndim, nb_spots):
    """Generate spots coordinates in 2-d or 3-d.

    Parameters
    ----------
    frame_shape : Tuple[int or float] or List[int of float]
        Shape (z, y, x) or (y, x) of the image to simulate.
    ndim : int
        Number of dimensions of the simulated image (2 or 3).
    nb_spots : int
        Number of spots to simulate.

    Returns
    -------
    positions_z : np.ndarray, np.int64
        Array of coordinates along the z axis, or None.
    positions_y : np.ndarray, np.int64
        Array of coordinates along the y axis.
    positions_x : np.ndarray, np.int64
        Array of coordinates along the x axis.

    """
    # simulate positions
    positions_z = None
    if ndim == 3:
        positions_z = np.random.uniform(0, frame_shape[0], size=nb_spots)
    positions_y = np.random.uniform(0, frame_shape[ndim - 2], size=nb_spots)
    positions_x = np.random.uniform(0, frame_shape[ndim - 1], size=nb_spots)

    # cast coordinates
    if ndim == 3:
        positions_z = positions_z.astype(np.int64)
    positions_y = positions_y.astype(np.int64)
    positions_x = positions_x.astype(np.int64)

    return positions_z, positions_y, positions_x


def _get_sigma(ndim, sigma_z, sigma_yx, random_sigma, random_level, nb_spots):
    """Get standard deviations of the gaussians.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the simulated image (2 or 3).
    sigma_z : int, float or None
        Standard deviation of the gaussian along the z axis, in nanometer. If
        None, we consider a 2-d image.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in nanometer.
    random_sigma : bool
        Make sigmas follow a tight normal distribution around the provided
        sigma values.
    random_level : float
        Margin allowed to change gaussian parameters provided. The formula
        used is margin = parameter * random_level
    nb_spots : int
        Number of spots to simulate.

    Returns
    -------
    sigmas_z : np.ndarray, np.float64
        Array of standard deviation along the z axis or None.
    sigmas_yx : np.ndarray, np.float64
        Array of standard deviation along the y or x axis.

    """
    # generate sigma values
    sigmas_z = None
    if ndim == 3:
        if not random_sigma:
            scale = 0
        else:
            scale = sigma_z * random_level
        sigmas_z = np.random.normal(loc=sigma_z, scale=scale, size=nb_spots)
    if not random_sigma:
        scale = 0
    else:
        scale = sigma_yx * random_level
    sigmas_yx = np.random.normal(loc=sigma_yx, scale=scale, size=nb_spots)

    return sigmas_z, sigmas_yx


def _get_amplitude(amplitude, random_amplitude, random_level, nb_spots):
    """Get amplitude of the simulated gaussians.

    Parameters
    ----------
    amplitude : int or float
        Amplitude of the gaussians.
    random_amplitude : bool
        Make amplitudes follow a uniform distribution around the provided
        amplitude value.
    random_level : float
        Margin allowed to change gaussian parameters provided. The formula
        used is margin = parameter * random_level
    nb_spots : int
        Number of spots to simulate.

    Returns
    -------
    amplitudes : np.ndarray, np.float64
        Array of gaussian amplitudes.

    """
    # generate amplitude values
    if random_amplitude:
        scale = amplitude * random_level
    else:
        scale = 0
    limit_down = amplitude - scale
    limit_up = amplitude + scale
    amplitudes = np.random.uniform(limit_down, limit_up, size=nb_spots)

    return amplitudes
