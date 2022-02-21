# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate spot coordinates and localization patterns.
"""

import numpy as np

import bigfish.stack as stack
import bigfish.multistack as multistack

from .utils import build_template


# ### Spots coordinates ###

def simulate_ground_truth(
        ndim,
        n_spots,
        random_n_spots=False,
        n_clusters=0,
        random_n_clusters=False,
        n_spots_cluster=0,
        random_n_spots_cluster=False,
        centered_cluster=False,
        frame_shape=(128, 128),
        voxel_size=(100, 100),
        sigma=(150,  150),
        random_sigma=0.05,
        amplitude=5000,
        random_amplitude=0.05,
        probability_map=None):
    """Simulate ground truth information about the simulated spots like their
    coordinates, standard deviations and amplitude.

    Parameters
    ----------
    ndim : {2, 3}
        Number of dimension to consider for the simulation.
    n_spots : int
        Expected number of spots to simulate.
    random_n_spots : bool, default=False
        Make the number of spots follow a Poisson distribution with
        expectation `n_spots`, instead of a constant predefined value.
    n_clusters : int
        Expected number of clusters to simulate.
    random_n_clusters : bool, default=False
        Make the number of clusters follow a Poisson distribution with
        expectation n_clusters, instead of a constant predefined value.
    n_spots_cluster : int
        Expected number of spots to simulate per cluster.
    random_n_spots_cluster : bool, default=False
        Make the number of spots follow a Poisson distribution with
        expectation `n_spots_cluster`, instead of a constant predefined value.
    centered_cluster : bool, default=False
        Center the simulated cluster. Only used if one cluster is simulated.
    frame_shape : tuple or list, default=(128, 128)
        Shape (z, y, x) or (y, x) of the image to simulate.
    voxel_size : int or float or tuple or list, default=(100, 100)
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    sigma : int or float or tuple or list, default=(150, 150)
        Standard deviation of the spot, in nanometer. One value per spatial
        dimension (zyx or yx dimensions). If it's a scalar, the same value is
        applied to every dimensions.
    random_sigma : int or float, default=0.05
        Sigmas follow a normal distribution around the provided sigma values.
        The scale used is:

        .. math::
            \\mbox{scale} = \\mbox{sigma} * \\mbox{random_sigma}
    amplitude : int or float, default=5000
        Intensity of the spot.
    random_amplitude : int or float, default=0.05
        Margin allowed around the amplitude value. The formula used is:

        .. math::
            \\mbox{margin} = \\mbox{amplitude} * \\mbox{random_amplitude}
    probability_map : np.ndarray, np.float32, optional
        Probability map to sample spots coordinates with shape (z, y, x).

    Returns
    -------
    ground_truth : np.ndarray, np.float64
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4). Columns
        are:

        * Coordinate along the z axis (optional).
        * Coordinate along the y axis.
        * Coordinate along the x axis.
        * Standard deviation of the spot along the z axis (optional).
        * Standard deviation of the spot in the yx plan.
        * Intensity of the spot.

    """
    # check parameters
    stack.check_parameter(
        ndim=int,
        n_spots=int,
        random_n_spots=bool,
        n_clusters=int,
        random_n_clusters=bool,
        n_spots_cluster=int,
        random_n_spots_cluster=bool,
        centered_cluster=bool,
        frame_shape=(tuple, list),
        voxel_size=(int, float, tuple, list),
        sigma=(int, float, tuple, list),
        random_sigma=(int, float),
        amplitude=(int, float),
        random_amplitude=(int, float))
    if probability_map is not None:
        stack.check_array(
            probability_map,
            ndim=[2, 3],
            dtype=[np.float32, np.float64])

    # check consistency between parameters
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if isinstance(sigma, (tuple, list)):
        if len(sigma) != ndim:
            raise ValueError(
                "'sigma' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        sigma = (sigma,) * ndim
    if len(frame_shape) != ndim:
        raise ValueError("'frame_shape' should have {0} elements, not {0}."
                         .format(ndim, len(frame_shape)))
    if probability_map is not None:
        if tuple(frame_shape) != probability_map.shape:
            raise ValueError(
                "Shape of 'probability_map' ({0}) does not match with "
                "provided 'frame_shape' ({1}).".format(probability_map.shape,
                                                       tuple(frame_shape)))

    # generate number of spots to simulate
    nb_spots = _get_nb_spots(n=n_spots, random_n=random_n_spots)

    # generate clusters
    (positions_z_clusters, positions_y_clusters, positions_x_clusters,
     remaining_spots) = _get_clusters(
        frame_shape=frame_shape,
        ndim=ndim,
        nb_spots=nb_spots,
        n_clusters=n_clusters,
        random_n_clusters=random_n_clusters,
        n_spots_cluster=n_spots_cluster,
        random_n_spots_cluster=random_n_spots_cluster,
        voxel_size=voxel_size,
        sigma=sigma,
        centered=centered_cluster,
        probability_map=probability_map)

    # simulate positions
    (positions_z_spots, positions_y_spots,
     positions_x_spots) = _get_spots_coordinates(
        frame_shape=frame_shape,
        ndim=ndim,
        nb_spots=remaining_spots,
        probability_map=probability_map)

    # merge coordinates
    if ndim == 3:
        positions_z = np.concatenate((positions_z_clusters, positions_z_spots))
    else:
        positions_z = None
    positions_y = np.concatenate((positions_y_clusters, positions_y_spots))
    positions_x = np.concatenate((positions_x_clusters, positions_x_spots))

    # generate sigma values
    sigmas_z, sigmas_yx = _get_sigma(
        ndim=ndim,
        sigma=sigma,
        random_sigma=random_sigma,
        nb_spots=nb_spots)

    # generate amplitude values
    amplitudes = _get_amplitude(
        amplitude=amplitude,
        random_amplitude=random_amplitude,
        nb_spots=nb_spots)

    # stack and format ground truth
    if ndim == 3:
        ground_truth = np.stack((
            positions_z, positions_y, positions_x,
            sigmas_z, sigmas_yx,
            amplitudes)).T
    else:
        ground_truth = np.stack((
            positions_y, positions_x,
            sigmas_yx,
            amplitudes)).T
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


def _get_clusters(
        frame_shape,
        ndim,
        nb_spots,
        n_clusters,
        random_n_clusters,
        n_spots_cluster,
        random_n_spots_cluster,
        voxel_size,
        sigma,
        centered=False,
        probability_map=None):
    """Generate number of clusters and coordinates for clustered spots.

    Parameters
    ----------
    frame_shape : Tuple or List
        Shape (z, y, x) or (y, x) of the image to simulate.
    ndim : int
        Number of dimensions of the simulated image (2 or 3).
    nb_spots : int
        Total number of spots to simulate in the image (clustered or not).
    n_clusters : int
        Expected number of clusters to simulate.
    random_n_clusters : bool
        Make the number of clusters follow a Poisson distribution with
        expectation `n_clusters`, instead of a constant predefined value.
    n_spots_cluster : int
        Expected number of spots to simulate per cluster.
    random_n_spots_cluster : bool
        Make the number of spots follow a Poisson distribution with
        expectation `n_spots_cluster`, instead of a constant predefined value.
    voxel_size : tuple or list
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions).
    sigma : tuple or list
        Standard deviation of the spot, in nanometer. One value per spatial
        dimension (zyx or yx dimensions).
    centered : bool, default=False
        Center the simulated cluster. Only used if one cluster is simulated.
    probability_map : np.ndarray, np.float32, optional
        Array of probability, with shape (z, y, x) or (y, x). Sum to one.

    Returns
    -------
    positions_z : np.ndarray, np.int64 or None
        Array of coordinates along the z axis or None.
    positions_y : np.ndarray, np.int64
        Array of coordinates along the y axis.
    positions_x : np.ndarray, np.int64
        Array of coordinates along the x axis.
    remaining_spots : int
        Remaining spots to simulate in the image.

    """
    # generate number of clusters to simulate
    nb_clusters = _get_nb_spots(n=n_clusters, random_n=random_n_clusters)

    # no cluster to simulate
    if nb_clusters == 0:
        positions_z = np.array([], dtype=np.int64).reshape((0,))
        positions_y = np.array([], dtype=np.int64).reshape((0,))
        positions_x = np.array([], dtype=np.int64).reshape((0,))
        return positions_z, positions_y, positions_x, nb_spots

    # multiple clusters can't be simulated in the center of the frame
    if nb_clusters > 1:
        centered = False

    # simulate cluster centers
    if probability_map is not None:
        sample = _sample_coordinates(nb_clusters, probability_map)
        center_cluster_z = None
        if ndim == 3:
            center_cluster_z = sample[:, 0]
        center_cluster_y = sample[:, ndim - 2]
        center_cluster_x = sample[:, ndim - 1]
    elif centered:
        center_cluster_z = None
        if ndim == 3:
            center_cluster_z = np.array(frame_shape[0] / 2, dtype=np.int64)
            center_cluster_z = np.reshape(center_cluster_z, -1)
        center_cluster_y = np.array(frame_shape[ndim - 2] / 2, dtype=np.int64)
        center_cluster_y = np.reshape(center_cluster_y, -1)
        center_cluster_x = np.array(frame_shape[ndim - 1] / 2, dtype=np.int64)
        center_cluster_x = np.reshape(center_cluster_x, -1)
    else:
        center_cluster_z = None
        if ndim == 3:
            center_cluster_z = np.random.uniform(
                0, frame_shape[0], size=nb_clusters)
        center_cluster_y = np.random.uniform(
            0, frame_shape[ndim - 2], size=nb_clusters)
        center_cluster_x = np.random.uniform(
            0, frame_shape[ndim - 1], size=nb_clusters)

    # get spots coordinates per cluster
    remaining_spots = nb_spots
    if ndim == 3:
        positions_z = []
    else:
        positions_z = None
    positions_y = []
    positions_x = []
    for i in range(nb_clusters):

        # get number of spots
        nb_spots_cluster = _get_nb_spots(
            n=n_spots_cluster, random_n=random_n_spots_cluster)
        nb_spots_cluster = min(nb_spots_cluster, remaining_spots)
        remaining_spots -= nb_spots_cluster

        # get spots coordinates
        scale_z = None
        if ndim == 3:
            spots_scale_z = sigma[0] / voxel_size[0]
            scale_z = spots_scale_z * 0.2 * nb_spots_cluster
        spots_scale_yx = sigma[-1] / voxel_size[-1]
        scale_yx = spots_scale_yx * 0.2 * nb_spots_cluster
        if ndim == 3:
            rho_z = np.abs(np.random.normal(
                loc=0.0, scale=scale_z, size=nb_spots_cluster))
            rho_yx = np.abs(np.random.normal(
                loc=0.0, scale=scale_yx, size=nb_spots_cluster))
            theta = np.random.uniform(0, np.pi, nb_spots_cluster)
            phi = np.random.uniform(0, 2 * np.pi, nb_spots_cluster)
            z = center_cluster_z[i] + rho_z * np.cos(theta)
            positions_z.append(z)
            y = center_cluster_y[i] + rho_yx * np.sin(phi) * np.sin(theta)
            positions_y.append(y)
            x = center_cluster_x[i] + rho_yx * np.cos(phi) * np.sin(theta)
            positions_x.append(x)

        else:
            rho_yx = np.random.normal(
                loc=0.0, scale=scale_yx, size=nb_spots_cluster)
            phi = np.random.uniform(-np.pi, np.pi, nb_spots_cluster)
            y = center_cluster_y[i] + rho_yx * np.sin(phi)
            positions_y.append(y)
            x = center_cluster_x[i] + rho_yx * np.cos(phi)
            positions_x.append(x)

    # concatenate and cast coordinates
    if ndim == 3:
        positions_z = np.concatenate(positions_z).astype(np.int64)
    positions_y = np.concatenate(positions_y).astype(np.int64)
    positions_x = np.concatenate(positions_x).astype(np.int64)

    # filter out spots incorrectly simulated
    mask_y = (positions_y >= 0) & (positions_y < frame_shape[ndim - 2])
    mask_x = (positions_x >= 0) & (positions_x < frame_shape[ndim - 1])
    if ndim == 3:
        mask_z = (positions_z >= 0) & (positions_z < frame_shape[0])
        mask = mask_z & mask_y & mask_x
        positions_z = positions_z[mask]
        positions_y = positions_y[mask]
        positions_x = positions_x[mask]
    else:
        mask = mask_y & mask_x
        positions_y = positions_y[mask]
        positions_x = positions_x[mask]
    if probability_map is not None:
        if ndim == 3:
            mask = probability_map[positions_z, positions_y, positions_x]
            mask = mask > 0.
            positions_z = positions_z[mask]
            positions_y = positions_y[mask]
            positions_x = positions_x[mask]
        else:
            mask = probability_map[positions_y, positions_x]
            mask = mask > 0.
            positions_y = positions_y[mask]
            positions_x = positions_x[mask]

    # compute remaining spots
    remaining_spots = nb_spots - mask.sum()

    return positions_z, positions_y, positions_x, remaining_spots


def _get_spots_coordinates(frame_shape, ndim, nb_spots, probability_map=None):
    """Generate spots coordinates in 2-d or 3-d.

    Parameters
    ----------
    frame_shape : tuple or list
        Shape (z, y, x) or (y, x) of the image to simulate.
    ndim : int
        Number of dimensions of the simulated image (2 or 3).
    nb_spots : int
        Number of spots to simulate.
    probability_map : np.ndarray, np.float32, optional
        Array of probability, with shape (z, y, x) or (y, x). Sum to one.

    Returns
    -------
    positions_z : np.ndarray, np.int64 or None
        Array of coordinates along the z axis, or None.
    positions_y : np.ndarray, np.int64
        Array of coordinates along the y axis.
    positions_x : np.ndarray, np.int64
        Array of coordinates along the x axis.

    """
    # simulate positions from a probability map
    if probability_map is not None:
        sample = _sample_coordinates(
            n=nb_spots, probability_map=probability_map)
        positions_z = None
        if ndim == 3:
            positions_z = sample[:, 0]
        positions_y = sample[:, ndim - 2]
        positions_x = sample[:, ndim - 1]

    # simulate positions from scratch
    else:
        positions_z = None
        if ndim == 3:
            positions_z = np.random.randint(0, frame_shape[0], nb_spots)
        positions_y = np.random.randint(0, frame_shape[ndim - 2], nb_spots)
        positions_x = np.random.randint(0, frame_shape[ndim - 1], nb_spots)

    # filter out spots incorrectly simulated
    if probability_map is not None:
        if ndim == 3:
            mask = probability_map[positions_z, positions_y, positions_x]
            mask = mask > 0.
            positions_z = positions_z[mask]
            positions_y = positions_y[mask]
            positions_x = positions_x[mask]
        else:
            mask = probability_map[positions_y, positions_x]
            mask = mask > 0.
            positions_y = positions_y[mask]
            positions_x = positions_x[mask]

    return positions_z, positions_y, positions_x


def _get_sigma(ndim, sigma, random_sigma, nb_spots):
    """Get standard deviations of the gaussians.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the simulated image (2 or 3).
    sigma : tuple or list
        Standard deviation of the spot, in nanometer. One value per spatial
        dimension (zyx or yx dimensions).
    random_sigma : int or float
        Sigmas follow a normal distribution around the provided sigma values.
        The scale used is:

        .. math::
            \\mbox{scale} = \\mbox{sigma} * \\mbox{random_sigma}
    nb_spots : int
        Number of spots to simulate.

    Returns
    -------
    sigmas_z : np.ndarray, np.float64 or None
        Array of standard deviation along the z axis or None.
    sigmas_yx : np.ndarray, np.float64
        Array of standard deviation along the y or x axis.

    """
    # generate sigma values
    sigmas_z = None
    scale = sigma[-1] * random_sigma
    sigmas_yx = np.random.normal(loc=sigma[-1], scale=scale, size=nb_spots)
    sigmas_yx[sigmas_yx < 1] = 1.
    if ndim == 3:
        scale = sigma[0] * random_sigma
        sigmas_z = np.random.normal(loc=sigma[0], scale=scale, size=nb_spots)
        sigmas_z[sigmas_z < 1] = 1.
    return sigmas_z, sigmas_yx


def _get_amplitude(amplitude, random_amplitude, nb_spots):
    """Get amplitude of the simulated gaussians.

    Parameters
    ----------
    amplitude : int or float
        Intensity of the spot.
    random_amplitude : int or float
        Margin allowed around the amplitude value. The formula used is:

        .. math::
            \\mbox{margin} = \\mbox{amplitude} * \\mbox{random_amplitude}
    nb_spots : int
        Number of spots to simulate.

    Returns
    -------
    amplitudes : np.ndarray, np.float64
        Array of spot amplitudes.

    """
    # generate amplitude values
    margin = amplitude * random_amplitude
    limit_down = amplitude - margin
    limit_up = amplitude + margin
    amplitudes = np.random.uniform(limit_down, limit_up, size=nb_spots)

    return amplitudes


def _sample_coordinates(n, probability_map):
    """Randomly sample coordinates in 2-d or 3-d,according to a probability
    map.

    Parameters
    ----------
    n : int
        Number of coordinates to sample.
    probability_map : np.ndarray, np.float32
        Array of probability, with shape (z, y, x) or (y, x). Sum to one.

    Returns
    -------
    sample : np.ndarray, np.int64
        Array of coordinates with shape (n, 3) or (n, 2).

    """
    # get frame dimension
    ndim = probability_map.ndim

    # get frame shape
    if ndim == 3:
        z_size, y_size, x_size = probability_map.shape
        z = np.linspace(0, z_size - 1, z_size)
    else:
        y_size, x_size = probability_map.shape
        z_size = None
        z = None
    y = np.linspace(0, y_size - 1, y_size)
    x = np.linspace(0, x_size - 1, x_size)

    # get frame coordinates
    if ndim == 3:
        zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
        coord_matrix = np.stack([zz, yy, xx], axis=-1)
        coord = coord_matrix.reshape((z_size * y_size * x_size, 3))
    else:
        yy, xx = np.meshgrid(y, x, indexing="ij")
        coord_matrix = np.stack([yy, xx], axis=-1)
        coord = coord_matrix.reshape((y_size * x_size, 2))
    coord = coord.astype(np.int64)

    # get coordinate indices
    index_coord = np.array([i for i in range(coord.shape[0])])

    # format probability array
    probability_map = probability_map.ravel()

    # sample coordinates
    index_sample = np.random.choice(
        index_coord, size=n, replace=False, p=probability_map)
    sample = coord[index_sample]

    return sample


# ### Probability maps ###

def _get_random_probability_map(cell_mask):
    """Compute a probability map to sample a random pattern.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Binary mask of the cell surface with shape (z, y, x).

    Returns
    -------
    probability_map : np.ndarray, np.float32
        Probability map to sample spots coordinates with shape (z, y, x).

    """
    # check parameters
    stack.check_array(cell_mask, ndim=3, dtype=bool)

    # random probability map
    probability_map = cell_mask.copy().astype(np.float32)
    probability_map /= probability_map.sum()

    return probability_map


def _get_random_out_probability_map(cell_mask, nuc_mask):
    """Compute a probability map to sample a random pattern outside nucleus.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Binary mask of the cell surface with shape (z, y, x).
    nuc_mask : np.ndarray, bool
        Binary mask of the nucleus surface with shape (z, y, x).

    Returns
    -------
    probability_map : np.ndarray, np.float32
        Probability map to sample spots coordinates with shape (z, y, x).

    """
    # check parameters
    stack.check_array(cell_mask, ndim=3, dtype=bool)
    stack.check_array(nuc_mask, ndim=3, dtype=bool)

    # random out probability map
    probability_map = cell_mask.copy().astype(np.float32)
    probability_map[nuc_mask] = 0.
    probability_map /= probability_map.sum()

    return probability_map


def _get_random_in_probability_map(cell_mask, nuc_mask):
    """Compute a probability map to sample a random pattern inside nucleus.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Binary mask of the cell surface with shape (z, y, x).
    nuc_mask : np.ndarray, bool
        Binary mask of the nucleus surface with shape (z, y, x).

    Returns
    -------
    probability_map : np.ndarray, np.float32
        Probability map to sample spots coordinates with shape (z, y, x).

    """
    # check parameters
    stack.check_array(cell_mask, ndim=3, dtype=bool)
    stack.check_array(nuc_mask, ndim=3, dtype=bool)

    # random in probability map
    probability_map = nuc_mask.copy().astype(np.float32)
    probability_map /= probability_map.sum()

    return probability_map


def _get_nuclear_edge_probability_map(cell_mask, nuc_map):
    """Compute a probability map to sample a nuclear edge pattern.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Binary mask of the cell surface with shape (z, y, x).
    nuc_map : np.ndarray, np.float32
        Distance map from the nucleus edge with shape (z, y, x).

    Returns
    -------
    probability_map : np.ndarray, np.float32
        Probability map to sample spots coordinates with shape (z, y, x).

    """
    # check parameters
    stack.check_array(cell_mask, ndim=3, dtype=bool)
    stack.check_array(nuc_map, ndim=3, dtype=[np.float32,  np.float64])

    # nuclear edge probability map
    probability_map = nuc_map.copy().astype(np.float32)
    probability_map[nuc_map > 2.] = 0.
    probability_map[nuc_map <= 2.] = 1.
    probability_map[~cell_mask] = 0.
    probability_map /= probability_map.sum()

    return probability_map


def _get_perinuclear_probability_map(cell_mask, cell_map, nuc_mask):
    """Compute a probability map to sample a perinuclear pattern.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Binary mask of the cell surface with shape (z, y, x).
    cell_map : np.ndarray, np.float32
        Distance map from the cell edge with shape (z, y, x).
    nuc_mask : np.ndarray, bool
        Binary mask of the nucleus surface with shape (z, y, x).

    Returns
    -------
    probability_map : np.ndarray, np.float32
        Probability map to sample spots coordinates with shape (z, y, x).

    """
    # check parameters
    stack.check_array(cell_mask, ndim=3, dtype=bool)
    stack.check_array(cell_map, ndim=3, dtype=[np.float32, np.float64])
    stack.check_array(nuc_mask, ndim=3, dtype=bool)

    # perinuclear probability map
    probability_map = cell_map.copy().astype(np.float32)
    probability_map **= 2
    probability_map[nuc_mask] = 0.
    probability_map[~cell_mask] = 0.
    probability_map /= probability_map.sum()

    return probability_map


def _get_cell_edge_probability_map(cell_mask, cell_map, nuc_mask):
    """Compute a probability map to sample a cell edge pattern.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Binary mask of the cell surface with shape (z, y, x).
    cell_map : np.ndarray, np.float32
        Distance map from the cell edge with shape (z, y, x).
    nuc_mask : np.ndarray, bool
        Binary mask of the nucleus surface with shape (z, y, x).

    Returns
    -------
    probability_map : np.ndarray, np.float32
        Probability map to sample spots coordinates with shape (z, y, x).

    """
    # check parameters
    stack.check_array(cell_mask, ndim=3, dtype=bool)
    stack.check_array(cell_map, ndim=3, dtype=[np.float32, np.float64])
    stack.check_array(nuc_mask, ndim=3, dtype=bool)

    # cell edge probability map
    probability_map = cell_map.copy().astype(np.float32)
    probability_map[cell_map > 2.] = 0.
    probability_map[cell_map <= 2.] = 1.
    probability_map[~cell_mask] = 0.
    probability_map[nuc_mask] = 0.
    probability_map /= probability_map.sum()

    return probability_map


def _get_protrusion_probability_map(
        cell_mask,
        nuc_mask,
        protrusion_mask):
    """Compute a probability map to sample a protrusion pattern.

    Parameters
    ----------
    cell_mask : np.ndarray, bool
        Binary mask of the cell surface with shape (z, y, x).
    nuc_mask : np.ndarray, bool
        Binary mask of the nucleus surface with shape (z, y, x).
    protrusion_mask : np.ndarray, bool
        Binary mask of the protrusion surface with shape (y, x).

    Returns
    -------
    probability_map : np.ndarray, np.float32
        Probability map to sample spots coordinates with shape (z, y, x).

    """
    # check parameters
    stack.check_array(cell_mask, ndim=3, dtype=bool)
    stack.check_array(nuc_mask, ndim=3, dtype=bool)
    stack.check_array(protrusion_mask, ndim=2, dtype=bool)

    # protrusion probability map
    probability_map = [protrusion_mask] * cell_mask.shape[0]
    probability_map = np.stack(probability_map, axis=0)
    probability_map = probability_map.astype(np.float32)
    probability_map[nuc_mask] = 0.
    probability_map[~cell_mask] = 0.
    probability_map /= probability_map.sum()

    return probability_map


def build_probability_map(
        path_template_directory,
        i_cell=None,
        index_template=None,
        map_distribution="random",
        return_masks=False):
    """Build a template and its probability map to sample spot coordinates.

    Parameters
    ----------
    path_template_directory : str
        Path of the templates directory.
    i_cell : int, optional
        Template id to build (between 0 and 317). If None, a random template
        is built.
    index_template : pd.DataFrame, optional
        Dataframe with the templates metadata. If None, dataframe is load from
        'path_template_directory'. Columns are:

        * 'id' instance id.
        * 'shape' shape of the cell image (with the format '{z}_{y}_{x}').
        * 'protrusion_flag' presence or not of protrusion in  the instance.
    map_distribution : str, default='random'
        Probability distribution map to generate among 'random', 'random_out',
        'random_in', 'nuclear_edge', 'perinuclear', 'cell_edge' and
        'protrusion'.
    return_masks : bool, default=False
        Return cell and nucleus binary masks.

    Returns
    -------
    probability_map : np.ndarray, np.float32
        Probability map to sample spots coordinates with shape (z, y, x).
    cell_mask : np.ndarray, bool, optional
        Binary mask of the cell surface with shape (z, y, x).
    nuc_mask : np.ndarray, bool, optional
        Binary mask of the nucleus surface with shape (z, y, x).

    """
    # check parameter
    stack.check_parameter(
        map_distribution=str,
        return_masks=bool)
    if map_distribution not in ["random", "random_out", "random_in",
                                "nuclear_edge", "perinuclear", "cell_edge",
                                "protrusion"]:
        raise ValueError("Probability maps available are: 'random', "
                         "'random_out', 'random_in', 'nuclear_edge', "
                         "'perinuclear', 'cell_edge', 'protrusion'. Not {0}."
                         .format(map_distribution))

    # set protrusion flag
    if map_distribution == "protrusion":
        protrusion = True
    else:
        protrusion = None

    # build template
    (cell_mask, cell_map, nuc_mask, nuc_map, protrusion_mask,
     protrusion_map) = build_template(
        path_template_directory=path_template_directory,
        i_cell=i_cell,
        index_template=index_template,
        protrusion=protrusion)

    # get probability map
    probability_map = None
    if map_distribution == "random":
        probability_map = _get_random_probability_map(cell_mask)
    elif map_distribution == "random_out":
        probability_map = _get_random_out_probability_map(cell_mask, nuc_mask)
    elif map_distribution == "random_in":
        probability_map = _get_random_in_probability_map(cell_mask, nuc_mask)
    elif map_distribution == "nuclear_edge":
        probability_map = _get_nuclear_edge_probability_map(cell_mask, nuc_map)
    elif map_distribution == "perinuclear":
        probability_map = _get_perinuclear_probability_map(
            cell_mask, cell_map, nuc_mask)
    elif map_distribution == "cell_edge":
        probability_map = _get_cell_edge_probability_map(
            cell_mask, cell_map, nuc_mask)
    elif map_distribution == "protrusion":
        probability_map = _get_protrusion_probability_map(
            cell_mask, nuc_mask, protrusion_mask)

    if return_masks:
        return probability_map, cell_mask, nuc_mask
    else:
        return probability_map


# ### Localization patterns (coordinates only) ###

def simulate_localization_pattern(
        path_template_directory,
        n_spots,
        i_cell=None,
        index_template=None,
        pattern="random",
        proportion_pattern=0.5):
    """Simulate spot coordinates with a specific localization pattern from a
    cell template.

    Parameters
    ----------
    path_template_directory : str
        Path of the templates directory.
    n_spots : int
        Number of spots to simulate.
    i_cell : int, optional
        Template id to build (between 0 and 317). If None, a random template
        is built.
    index_template : pd.DataFrame, optional
        Dataframe with the templates metadata. If None, dataframe is load from
        'path_template_directory'. Columns are:

        * 'id' instance id.
        * 'shape' shape of the cell image (with the format '{z}_{y}_{x}').
        * 'protrusion_flag' presence or not of protrusion in  the instance.
    pattern : str, default='random'
        Spot localization pattern to simulate among 'random', 'foci',
        'intranuclear', 'nuclear_edge', 'perinuclear', 'cell_edge' and
        'protrusion'.
    proportion_pattern : int or float, default=0.5
        Proportion of spots to localize with the pattern. Value between 0 and 1
        (0 means random pattern and 1 a perfect pattern).

    Returns
    -------
    instance_coord : dict
        Dictionary with information about the cell:

        * `cell_id`: Unique id of the cell.
        * `bbox`: bounding box coordinates with the order (`min_y`, `min_x`,
          `max_y`, `max_x`).
        * `cell_coord`: boundary coordinates of the cell.
        * `cell_mask`: mask of the cell.
        * `nuc_coord`: boundary coordinates of the nucleus.
        * `nuc_mask`: mask of the nucleus.
        * `rna_coord`: rna spot coordinates.

    """
    # check parameter
    stack.check_parameter(
        n_spots=int,
        pattern=str,
        proportion_pattern=(int, float))
    if pattern not in ["random", "foci", "intranuclear", "nuclear_edge",
                       "perinuclear", "cell_edge", "protrusion"]:
        raise ValueError("Patterns available are: 'random', 'foci', "
                         "'intranuclear', 'nuclear_edge', 'perinuclear', "
                         "'cell_edge', 'protrusion'. Not {0}.".format(pattern))

    # simulate foci pattern (special case)
    if pattern == "foci":
        ground_truth, cell_mask, nuc_mask, ndim = _simulate_foci_pattern(
            path_template_directory=path_template_directory,
            n_spots=n_spots,
            i_cell=i_cell,
            index_template=index_template,
            proportion_pattern=proportion_pattern)

    # simulate any other pattern
    else:
        ground_truth, cell_mask, nuc_mask, ndim = _simulate_pattern_area(
            path_template_directory=path_template_directory,
            n_spots=n_spots,
            i_cell=i_cell,
            index_template=index_template,
            pattern=pattern,
            proportion_pattern=proportion_pattern)

    # get instance coordinates
    cell_mask_2d = cell_mask.max(axis=0).astype(np.uint8)
    nuc_mask_2d = nuc_mask.max(axis=0).astype(np.uint8)
    rna_coord = ground_truth[:, :ndim].copy()
    instance_coord = multistack.extract_cell(
        cell_label=cell_mask_2d,
        ndim=ndim,
        nuc_label=nuc_mask_2d,
        rna_coord=rna_coord)[0]

    return instance_coord


def _simulate_foci_pattern(
        path_template_directory,
        n_spots,
        i_cell,
        index_template,
        proportion_pattern):
    """Simulate foci localization pattern. Spots are localized uniformly in
    the cell, but clusters are simulated outside nucleus accordingly to the
    'strength' parameter.

    Parameters
    ----------
    path_template_directory : str
        Path of the templates directory.
    n_spots : int
        Number of spots to simulate.
    i_cell : int
        Template id to build (between 0 and 317). If None, a random template
        is built.
    index_template : pd.DataFrame
        Dataframe with the templates metadata. If None, dataframe is load from
        'path_template_directory'. Columns are:

        * 'id' instance id.
        * 'shape' shape of the cell image (with the format '{z}_{y}_{x}').
        * 'protrusion_flag' presence or not of protrusion in  the instance.
    proportion_pattern : int or float
        Proportion of spots to localize with the pattern. Value between 0 and 1
        (0 means random pattern and 1 a perfect pattern).

    Returns
    -------
    ground_truth : np.ndarray, np.float64
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4). Columns
        are:

        * Coordinate along the z axis (optional).
        * Coordinate along the y axis.
        * Coordinate along the x axis.
        * Standard deviation of the spot along the z axis (optional).
        * Standard deviation of the spot in the yx plan.
        * Intensity of the spot.
    cell_mask : np.ndarray, bool, optional
        Binary mask of the cell surface with shape (z, y, x).
    nuc_mask : np.ndarray, bool, optional
        Binary mask of the nucleus surface with shape (z, y, x).
    ndim : int
        Number of dimensions of the simulated image (2 or 3).

    """

    # build probability maps
    probability_map_random_out, cell_mask, nuc_mask = build_probability_map(
        path_template_directory=path_template_directory,
        i_cell=i_cell,
        index_template=index_template,
        map_distribution="random_out",
        return_masks=True)
    probability_map_random_in = build_probability_map(
        path_template_directory=path_template_directory,
        i_cell=i_cell,
        index_template=index_template,
        map_distribution="random_in")

    # set number and size of foci
    n_spots_per_foci = np.random.randint(5, 21)
    n_spots_foci = int(n_spots * proportion_pattern)
    n_foci = int(n_spots_foci / n_spots_per_foci)

    # balance proportion of spots simulated inside and outside nucleus
    volume_nuc = nuc_mask.sum()
    proportion_nuc = volume_nuc / cell_mask.sum()
    n_spots_nuc = int((n_spots - n_spots_foci) * proportion_nuc)
    n_spots_cell = n_spots - n_spots_nuc

    # simulate ground truth for clustered spots (outside nucleus)
    frame_shape = probability_map_random_out.shape
    ndim = len(frame_shape)
    voxel_size = (100,) * ndim
    sigma = (150,) * ndim
    ground_truth_outside = simulate_ground_truth(
        ndim=ndim,
        n_spots=n_spots_cell,
        n_clusters=n_foci,
        n_spots_cluster=n_spots_per_foci,
        random_n_spots_cluster=True,
        frame_shape=frame_shape,
        voxel_size=voxel_size,
        sigma=sigma,
        probability_map=probability_map_random_out)

    # simulate ground truth for random spots (inside nucleus)
    ground_truth_inside = simulate_ground_truth(
        ndim=ndim,
        n_spots=n_spots_nuc,
        frame_shape=frame_shape,
        voxel_size=voxel_size,
        sigma=sigma,
        probability_map=probability_map_random_in)

    # stack simulated spots
    ground_truth = np.concatenate(
        (ground_truth_outside, ground_truth_inside), axis=0)

    return ground_truth, cell_mask, nuc_mask, ndim


def _simulate_pattern_area(
        path_template_directory,
        n_spots,
        i_cell,
        index_template,
        pattern,
        proportion_pattern):
    """Simulate localization pattern defined as an oversampling of spots in a
    subcellular region.

    Parameters
    ----------
    path_template_directory : str
        Path of the templates directory.
    n_spots : int
        Number of spots to simulate.
    i_cell : int
        Template id to build (between 0 and 317). If None, a random template
        is built.
    index_template : pd.DataFrame
        Dataframe with the templates metadata. If None, dataframe is load from
        'path_template_directory'. Columns are:

        * 'id' instance id.
        * 'shape' shape of the cell image (with the format '{z}_{y}_{x}').
        * 'protrusion_flag' presence or not of protrusion in  the instance.
    pattern : str
        Spot localization pattern to simulate among 'intranuclear',
        'nuclear_edge', 'perinuclear', 'cell_edge', 'protrusion' and 'random'.
    proportion_pattern : int or float
        Proportion of spots to localize with the pattern. Value between 0 and 1
        (0 means random pattern and 1 a perfect pattern).

    Returns
    -------
    ground_truth : np.ndarray, np.float64
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4). Columns
        are:

        * Coordinate along the z axis (optional).
        * Coordinate along the y axis.
        * Coordinate along the x axis.
        * Standard deviation of the spot along the z axis (optional).
        * Standard deviation of the spot in the yx plan.
        * Intensity of the spot.
    cell_mask : np.ndarray, bool, optional
        Binary mask of the cell surface with shape (z, y, x).
    nuc_mask : np.ndarray, bool, optional
        Binary mask of the nucleus surface with shape (z, y, x).
    ndim : int
        Number of dimensions of the simulated image (2 or 3).

    """
    # define probability map needed
    if pattern == "intranuclear":
        map_distribution = "random_in"
    elif pattern == "nuclear_edge":
        map_distribution = "nuclear_edge"
    elif pattern == "perinuclear":
        map_distribution = "perinuclear"
    elif pattern == "cell_edge":
        map_distribution = "cell_edge"
    elif pattern == "protrusion":
        map_distribution = "protrusion"
    else:
        map_distribution = "random"

    # build probability map with the targeted region
    probability_map_pattern, cell_mask, nuc_mask = build_probability_map(
        path_template_directory=path_template_directory,
        i_cell=i_cell,
        index_template=index_template,
        map_distribution=map_distribution,
        return_masks=True)

    # no localization pattern
    if pattern == "random":

        # simulate ground truth for random spots
        frame_shape = cell_mask.shape
        ndim = len(frame_shape)
        voxel_size = (100,) * ndim
        sigma = (150,) * ndim
        ground_truth = simulate_ground_truth(
            ndim=ndim,
            n_spots=n_spots,
            frame_shape=frame_shape,
            voxel_size=voxel_size,
            sigma=sigma,
            probability_map=probability_map_pattern)

    # localization pattern
    else:

        # build probability map without the targeted region
        if pattern == "perinuclear":
            probability_map_nopattern = cell_mask.copy().astype(np.float32)
            probability_map_nopattern /= probability_map_nopattern.sum()
        else:
            probability_map_nopattern = cell_mask.copy().astype(np.float32)
            probability_map_nopattern[probability_map_pattern > 0] = 0.
            probability_map_nopattern /= probability_map_nopattern.sum()

        # Get the number of spots following the pattern
        n_spots_pattern = int(n_spots * proportion_pattern)
        n_spots_nopattern = n_spots - n_spots_pattern

        # simulate ground truth for spots in targeted region
        frame_shape = cell_mask.shape
        ndim = len(frame_shape)
        voxel_size = (100,) * ndim
        sigma = (150,) * ndim
        ground_truth_pattern = simulate_ground_truth(
            ndim=ndim,
            n_spots=n_spots_pattern,
            frame_shape=frame_shape,
            voxel_size=voxel_size,
            sigma=sigma,
            probability_map=probability_map_pattern)

        # simulate ground truth for random spots (outside targeted region)
        ground_truth_nopattern = simulate_ground_truth(
            ndim=ndim,
            n_spots=n_spots_nopattern,
            frame_shape=frame_shape,
            voxel_size=voxel_size,
            sigma=sigma,
            probability_map=probability_map_nopattern)

        # stack simulated spots
        ground_truth = np.concatenate(
            (ground_truth_pattern, ground_truth_nopattern), axis=0)

    return ground_truth, cell_mask, nuc_mask, ndim
