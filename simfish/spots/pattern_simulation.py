# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate spots patterns.
"""

import numpy as np
import bigfish.stack as stack

# TODO add a pattern with different densities per area


def simulate_ground_truth(n_spots=30, random_n_spots=False, n_clusters=0,
                          random_n_clusters=False, n_spots_cluster=0,
                          frame_shape=(128, 128), sigma_z=None, sigma_yx=150,
                          random_sigma=0.05, amplitude=5000,
                          random_amplitude=0.05):
    """ Simulate ground truth information about the simulated spots like their
    coordinates, standard deviations and amplitude.

    Parameters
    ----------
    n_spots : int
        Expected number of spots to simulate.
    random_n_spots : bool
        Make the number of spots follow a Poisson distribution with
        expectation n_spots, instead of a constant predefined value.
    n_clusters : int
        Expected number of clusters to simulate.
    random_n_clusters : bool
        Make the number of clusters follow a Poisson distribution with
        expectation n_clusters, instead of a constant predefined value.
    n_spots_cluster : int
        Expected number of spots to simulate per cluster.
    frame_shape : Tuple[int or float] or List[int of float]
        Shape (z, y, x) or (y, x) of the image to simulate.
    sigma_z : int, float or None
        Standard deviation of the gaussian along the z axis, in nanometer. If
        None, we consider a 2-d image.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in nanometer.
    random_sigma : int of float
        Sigmas follow a normal distribution around the provided sigma values.
        The scale used is scale = sigma_axis * random_sigma
    amplitude : int or float
        Amplitude of the gaussians.
    random_amplitude : int or float
        Margin allowed around the amplitude value. The formula used is
        margin = parameter * random_level.

    Returns
    -------
    ground_truth : np.ndarray, np.float64
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4).
        - coordinate_z (optional)
        - coordinate_y
        - coordinate_x
        - sigma_z (optional)
        - sigma_yx
        - amplitude

    """
    # check parameters
    stack.check_parameter(n_spots=int,
                          random_n_spots=bool,
                          n_clusters=int,
                          random_n_clusters=bool,
                          n_spots_cluster=int,
                          frame_shape=(tuple, list),
                          sigma_z=(int, float, type(None)),
                          sigma_yx=(int, float),
                          random_sigma=(int, float),
                          amplitude=(int, float),
                          random_amplitude=(int, float))

    # check dimensions
    ndim = len(frame_shape)
    if ndim not in [2, 3]:
        raise ValueError("'frame_shape' should have 2 or 3 elements, not {0}."
                         .format(ndim))
    if ndim == 3 and sigma_z is None:
        raise ValueError("Frame to simulate has 3 dimensions but 'sigma_z' is "
                         "missing.")

    # generate number of spots to simulate
    nb_spots = _get_nb_spots(n_spots, random_n_spots)

    # generate clusters
    (positions_z_clusters, positions_y_clusters, positions_x_clusters,
     remaining_spots) = _get_clusters(
        frame_shape, ndim, nb_spots, n_clusters, random_n_clusters,
        n_spots_cluster)

    # simulate positions
    (positions_z_spots, positions_y_spots,
     positions_x_spots) = _get_spots_coordinates(
        frame_shape, ndim, remaining_spots)

    # merge coordinates
    if ndim == 3:
        positions_z = np.concatenate((positions_z_clusters, positions_z_spots))
    else:
        positions_z = None
    positions_y = np.concatenate((positions_y_clusters, positions_y_spots))
    positions_x = np.concatenate((positions_x_clusters, positions_x_spots))

    # generate sigma values
    sigmas_z, sigmas_yx = _get_sigma(ndim, sigma_z, sigma_yx, random_sigma,
                                     nb_spots)

    # generate amplitude values
    amplitudes = _get_amplitude(amplitude, random_amplitude, nb_spots)

    # stack and format ground truth
    if ndim == 3:
        ground_truth = np.stack((positions_z, positions_y, positions_x,
                                 sigmas_z, sigmas_yx, amplitudes)).T
    else:
        ground_truth = np.stack((positions_y, positions_x,
                                 sigmas_yx, amplitudes)).T
    ground_truth = ground_truth.astype(np.float64)

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


def _get_clusters(frame_shape, ndim, nb_spots, n_clusters, random_n_clusters,
                  n_spots_cluster):
    """Generate number of clusters and coordinates for clustered spots.

    Parameters
    ----------
    frame_shape : Tuple[int or float] or List[int of float]
        Shape (z, y, x) or (y, x) of the image to simulate.
    ndim : int
        Number of dimensions of the simulated image (2 or 3).
    nb_spots : int
        Total number of spots to simulate in the image (clustered or not).
    n_clusters : int
        Expected number of clusters to simulate.
    random_n_clusters : bool
        Make the number of clusters follow a Poisson distribution with
        expectation n_clusters, instead of a constant predefined value.
    n_spots_cluster : int
        Expected number of clusters to simulate per cluster.

    Returns
    -------
    positions_z : np.ndarray, np.int64
        Array of coordinates along the z axis, or None.
    positions_y : np.ndarray, np.int64
        Array of coordinates along the y axis.
    positions_x : np.ndarray, np.int64
        Array of coordinates along the x axis.
    remaining_spots : int
        Remaining spots to simulate in the image.

    """
    # generate number of clusters to simulate
    nb_clusters = _get_nb_spots(n_clusters, random_n_clusters)

    if nb_clusters == 0:
        positions_z = np.array([], dtype=np.int64).reshape((0,))
        positions_y = np.array([], dtype=np.int64).reshape((0,))
        positions_x = np.array([], dtype=np.int64).reshape((0,))
        return positions_z, positions_y, positions_x, nb_spots

    # get cluster center
    center_cluster_z = None
    if ndim == 3:
        center_cluster_z = np.random.uniform(0, frame_shape[0],
                                             size=nb_clusters)
    center_cluster_y = np.random.uniform(0, frame_shape[ndim - 2],
                                         size=nb_clusters)
    center_cluster_x = np.random.uniform(0, frame_shape[ndim - 1],
                                         size=nb_clusters)

    # get spots coordinates per cluster
    remaining_spots = nb_spots
    if ndim == 3:
        positions_z = []
    else:
        positions_z = None
    positions_y = []
    positions_x = []
    for i_cluster in range(nb_clusters):

        # get number of spots
        nb_spots_cluster = _get_nb_spots(n_spots_cluster, True)
        nb_spots_cluster = min(nb_spots_cluster, remaining_spots)
        remaining_spots -= nb_spots_cluster

        # get spots coordinates
        random_scale = np.random.uniform(0, 0.5, 1)
        scale = 1.0 + random_scale * nb_spots_cluster
        if ndim == 3:
            rho = np.abs(np.random.normal(loc=0.0, scale=scale,
                                          size=nb_spots_cluster))
            theta = np.random.uniform(0, np.pi, nb_spots_cluster)
            phi = np.random.uniform(0, 2 * np.pi, nb_spots_cluster)

            z = center_cluster_z[i_cluster] + rho * np.cos(theta)
            positions_z.append(z)

            y = center_cluster_y[i_cluster] + rho * np.sin(phi) * np.sin(theta)
            positions_y.append(y)

            x = center_cluster_x[i_cluster] + rho * np.cos(phi) * np.sin(theta)
            positions_x.append(x)

        else:
            rho = np.random.normal(loc=0.0, scale=scale, size=nb_spots_cluster)
            phi = np.random.uniform(-np.pi, np.pi, nb_spots_cluster)

            y = center_cluster_y[i_cluster] + rho * np.sin(phi)
            positions_y.append(y)

            x = center_cluster_x[i_cluster] + rho * np.cos(phi)
            positions_x.append(x)

    # concatenate and cast coordinates
    positions_y = np.concatenate(positions_y).astype(np.int64)
    mask_y = (positions_y >= 0) & (positions_y < frame_shape[ndim - 2])
    positions_x = np.concatenate(positions_x).astype(np.int64)
    mask_x = (positions_x >= 0) & (positions_x < frame_shape[ndim - 1])
    if ndim == 3:
        positions_z = np.concatenate(positions_z).astype(np.int64)
        mask_z = (positions_z >= 0) & (positions_z < frame_shape[0])
        mask = mask_z & mask_y & mask_x
        positions_z = positions_z[mask]
        positions_y = positions_y[mask]
        positions_x = positions_x[mask]

    else:
        mask = mask_y & mask_x
        positions_y = positions_y[mask]
        positions_x = positions_x[mask]

    # compute remaining spots
    remaining_spots = nb_spots - mask.sum()

    return positions_z, positions_y, positions_x, remaining_spots


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


def _get_sigma(ndim, sigma_z, sigma_yx, random_sigma, nb_spots):
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
    random_sigma : int of float
        Sigmas follow a normal distribution around the provided sigma values.
        The scale used is scale = sigma_axis * random_sigma
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
        scale = sigma_z * random_sigma
        sigmas_z = np.random.normal(loc=sigma_z, scale=scale, size=nb_spots)
        sigmas_z[sigmas_z < 1] = 1.
    scale = sigma_yx * random_sigma
    sigmas_yx = np.random.normal(loc=sigma_yx, scale=scale, size=nb_spots)
    sigmas_yx[sigmas_yx < 1] = 1.

    return sigmas_z, sigmas_yx


def _get_amplitude(amplitude, random_amplitude, nb_spots):
    """Get amplitude of the simulated gaussians.

    Parameters
    ----------
    amplitude : int or float
        Amplitude of the gaussians.
    random_amplitude : int or float
        Margin allowed around the amplitude value. The formula used is
        margin = parameter * random_level.
    nb_spots : int
        Number of spots to simulate.

    Returns
    -------
    amplitudes : np.ndarray, np.float64
        Array of gaussian amplitudes.

    """
    # generate amplitude values
    margin = amplitude * random_amplitude
    limit_down = amplitude - margin
    limit_up = amplitude + margin
    amplitudes = np.random.uniform(limit_down, limit_up, size=nb_spots)

    return amplitudes
