# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate RNA spots with gaussian functions.
"""

import numpy as np

import simfish.utils as utils

from scipy.special import erf

# TODO generate grid at nanometer scale (instead of 5nm currently)


# ### Gaussian computation ###

def precompute_erf(voxel_size_z=None, voxel_size_yx=100, sigma_z=None,
                   sigma_yx=200, grid_size=200):
    """Precompute different values for the erf with a nanometer resolution.

    Parameters
    ----------
    voxel_size_z : float or int or None
        Height of a voxel, in nanometer. If None, we consider a 2-d erf.
    voxel_size_yx : float or int
        size of a voxel, in nanometer.
    sigma_z : float or int or None
        Standard deviation of the gaussian along the z axis, in nanometer. If
        None, we consider a 2-d erf.
    sigma_yx : float or int
        Standard deviation of the gaussian along the yx axis, in nanometer.
    grid_size : int
        Size of the grid on which we precompute the erf, in pixel.

    Returns
    -------
    table_erf : Tuple[np.ndarray]
        Tuple with tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension. First column is the coordinate
        along the table dimension. Second column is the precomputed erf value.

    """
    # check parameters
    utils.check_parameter(voxel_size_z=(float, int, type(None)),
                          voxel_size_yx=(float, int),
                          sigma_z=(float, int, type(None)),
                          sigma_yx=(float, int),
                          grid_size=int)

    # build a grid with a spatial resolution of 1 nm and a size of
    # max_grid * resolution nm
    max_size_yx = np.ceil(grid_size * voxel_size_yx).astype(np.int64)
    yy = np.array([i for i in range(0, max_size_yx)])
    xx = np.array([i for i in range(0, max_size_yx)])
    mu_y, mu_x = 0, 0

    # compute erf values for this grid
    erf_y = _rescaled_erf(low=yy - voxel_size_yx/2,
                          high=yy + voxel_size_yx/2,
                          mu=mu_y,
                          sigma=sigma_yx)
    erf_x = _rescaled_erf(low=xx - voxel_size_yx/2,
                          high=xx + voxel_size_yx/2,
                          mu=mu_x,
                          sigma=sigma_yx)

    table_erf_y = np.array([yy, erf_y]).T
    table_erf_x = np.array([xx, erf_x]).T

    # precompute erf along z axis if needed
    if voxel_size_z is None or sigma_z is None:
        return table_erf_y, table_erf_x

    else:
        max_size_z = np.ceil(grid_size * voxel_size_z).astype(np.int64)
        zz = np.array([i for i in range(0, max_size_z)])
        mu_z = 0
        erf_z = _rescaled_erf(low=zz - voxel_size_z / 2,
                              high=zz + voxel_size_z / 2,
                              mu=mu_z,
                              sigma=sigma_z)
        table_erf_z = np.array([zz, erf_z]).T
        return table_erf_z, table_erf_y, table_erf_x


def _rescaled_erf(low, high, mu, sigma):
    """Rescaled the Error function along a specific axis.

    # TODO add equations

    Parameters
    ----------
    low : np.ndarray, np.float
        Lower bound of the voxel along a specific axis.
    high : np.ndarray, np.float
        Upper bound of the voxel along a specific axis.
    mu : int or float
        Estimated mean of the gaussian signal along a specific axis.
    sigma : int or float
        Estimated standard deviation of the gaussian signal along a specific
        axis.

    Returns
    -------
    rescaled_erf : np.ndarray, np.float
        Rescaled erf along a specific axis.

    """
    # TODO to speed up
    # compute erf and normalize it
    low_ = (low - mu) / (np.sqrt(2) * sigma)
    high_ = (high - mu) / (np.sqrt(2) * sigma)
    rescaled_erf = sigma * np.sqrt(np.pi / 2) * (erf(high_) - erf(low_))

    return rescaled_erf


def _gaussian_3d(grid, mu_z, mu_y, mu_x, sigma_z, sigma_yx, voxel_size_z,
                 voxel_size_yx, psf_amplitude, psf_background,
                 precomputed=None):
    """Compute the gaussian function over the grid 'xdata' representing a
    volume V with shape (V_z, V_y, V_x).

    # TODO add equations

    Parameters
    ----------
    grid : np.ndarray, np.int64
        Grid data to compute the gaussian function for different voxel within
        a volume V. In nanometer, with shape (3, V_z * V_y * V_x).
    mu_z : int
        Estimated mean of the gaussian signal along z axis, in nanometer.
    mu_y : int
        Estimated mean of the gaussian signal along y axis, in nanometer.
    mu_x : int
        Estimated mean of the gaussian signal along x axis, in nanometer.
    sigma_z : int or float
        Standard deviation of the gaussian along the z axis, in nanometer.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in nanometer.
    voxel_size_z : int or float
        Height of a voxel, in nanometer.
    voxel_size_yx : int or float
        size of a voxel, in nanometer.
    psf_amplitude : float
        Estimated pixel intensity of a spot.
    psf_background : float
        Estimated pixel intensity of the background.
    precomputed : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    values : np.ndarray, np.float
        Value of each voxel within the volume V according to the 3-d gaussian
        parameters. Shape (V_z * V_y * V_x,).

    """
    # TODO to speed up
    # get grid data to design a volume V
    meshgrid_z = grid[0]
    meshgrid_y = grid[1]
    meshgrid_x = grid[2]

    # use precomputed tables
    if precomputed is not None:
        # get tables
        table_erf_z = precomputed[0]
        table_erf_y = precomputed[1]
        table_erf_x = precomputed[2]

        # get indices for the tables
        i_z = np.abs(meshgrid_z - mu_z)
        i_y = np.abs(meshgrid_y - mu_y)
        i_x = np.abs(meshgrid_x - mu_x)

        # get precomputed values
        voxel_integral_z = table_erf_z[i_z, 1]
        voxel_integral_y = table_erf_y[i_y, 1]
        voxel_integral_x = table_erf_x[i_x, 1]

    # compute erf value
    else:
        # get voxel coordinates
        meshgrid_z_minus = meshgrid_z - voxel_size_z / 2
        meshgrid_z_plus = meshgrid_z + voxel_size_z / 2
        meshgrid_y_minus = meshgrid_y - voxel_size_yx / 2
        meshgrid_y_plus = meshgrid_y + voxel_size_yx / 2
        meshgrid_x_minus = meshgrid_x - voxel_size_yx / 2
        meshgrid_x_plus = meshgrid_x + voxel_size_yx / 2

        # compute gaussian function for each voxel (i, j, k) of volume V
        voxel_integral_z = _rescaled_erf(low=meshgrid_z_minus,
                                         high=meshgrid_z_plus,
                                         mu=mu_z,
                                         sigma=sigma_z)
        voxel_integral_y = _rescaled_erf(low=meshgrid_y_minus,
                                         high=meshgrid_y_plus,
                                         mu=mu_y,
                                         sigma=sigma_yx)
        voxel_integral_x = _rescaled_erf(low=meshgrid_x_minus,
                                         high=meshgrid_x_plus,
                                         mu=mu_x,
                                         sigma=sigma_yx)

    # compute 3-d gaussian values
    factor = psf_amplitude / (voxel_size_yx ** 2 * voxel_size_z)
    voxel_integral = voxel_integral_z * voxel_integral_y * voxel_integral_x
    values = psf_background + factor * voxel_integral

    return values


def _gaussian_2d(grid, mu_y, mu_x, sigma_yx, voxel_size_yx, psf_amplitude,
                 psf_background, precomputed=None):
    """Compute the gaussian function over the grid 'xdata' representing a
    surface S with shape (S_y, S_x).

    # TODO add equations

    Parameters
    ----------
    grid : np.ndarray, np.int64
        Grid data to compute the gaussian function for different voxel within
        a surface S. In nanometer, with shape (2, S_y * S_x).
    mu_y : int
        Estimated mean of the gaussian signal along y axis, in nanometer.
    mu_x : int
        Estimated mean of the gaussian signal along x axis, in nanometer.
    sigma_yx : int or float
        Standard deviation of the gaussian along the yx axis, in nanometer.
    voxel_size_yx : int or float
        size of a voxel, in nanometer.
    psf_amplitude : float
        Estimated pixel intensity of a spot.
    psf_background : float
        Estimated pixel intensity of the background.
    precomputed : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    values : np.ndarray, np.float
        Value of each voxel within the surface S according to the 2-d gaussian
        parameters. Shape (S_y * S_x,).

    """
    # TODO to speed up
    # get grid data to design a surface S
    meshgrid_y = grid[0]
    meshgrid_x = grid[1]

    # use precomputed tables
    if precomputed is not None:
        # get tables
        table_erf_y = precomputed[0]
        table_erf_x = precomputed[1]

        # get indices for the tables
        i_y = np.abs(meshgrid_y - mu_y)
        i_x = np.abs(meshgrid_x - mu_x)

        # get precomputed values
        voxel_integral_y = table_erf_y[i_y, 1]
        voxel_integral_x = table_erf_x[i_x, 1]

    # compute erf value
    else:
        # get voxel coordinates
        meshgrid_y_minus = meshgrid_y - voxel_size_yx / 2
        meshgrid_y_plus = meshgrid_y + voxel_size_yx / 2
        meshgrid_x_minus = meshgrid_x - voxel_size_yx / 2
        meshgrid_x_plus = meshgrid_x + voxel_size_yx / 2

        # compute gaussian function for each voxel (i, j) of surface S
        voxel_integral_y = _rescaled_erf(low=meshgrid_y_minus,
                                         high=meshgrid_y_plus,
                                         mu=mu_y,
                                         sigma=sigma_yx)
        voxel_integral_x = _rescaled_erf(low=meshgrid_x_minus,
                                         high=meshgrid_x_plus,
                                         mu=mu_x,
                                         sigma=sigma_yx)

    # compute 2-d gaussian values
    factor = psf_amplitude / (voxel_size_yx ** 2)
    voxel_integral = voxel_integral_y * voxel_integral_x
    values = psf_background + factor * voxel_integral

    return values


# ### Gaussian simulation ###

def add_spots(image, ground_truth, voxel_size_z=None, voxel_size_yx=100,
              precomputed_gaussian=None):
    """Simulate spots based on the ground truth coordinates.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (z, y, x) or (y, x).
    ground_truth : np.ndarray
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4).
        - coordinate_z (optional)
        - coordinate_y
        - coordinate_x
        - sigma_z (optional)
        - sigma_yx
        - amplitude
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we
        consider a 2-d image.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    precomputed_gaussian : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    image_with_spots : np.ndarray, np.uint
        A 3-d or 2-d image with simulated spots and shape (z, y, x) or (y, x).

    """
    # check parameters
    utils.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16])
    utils.check_array(ground_truth,
                      ndim=2,
                      dtype=np.float64)
    utils.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float))

    # check number of dimensions
    ndim = image.ndim
    if ndim == 3 and voxel_size_z is None:
        raise ValueError("Provided image has {0} dimensions but "
                         "'voxel_size_z' parameter is missing.".format(ndim))
    if ndim == 3 and ground_truth.shape[1] != 6:
        raise ValueError("Provided image has 3 dimensions but 'ground_truth' "
                         "has only {0} columns instead of 6."
                         .format(ground_truth.shape[1]))
    if ndim == 2:
        voxel_size_z = None

    # simulate and add 3-d spots...
    if image.ndim == 3:
        image_with_spots = _add_spot_3d(
            image,
            ground_truth,
            voxel_size_z,
            voxel_size_yx,
            precomputed_gaussian)

    # ... or 2-d spots
    else:
        image_with_spots = _add_spot_2d(
            image,
            ground_truth,
            voxel_size_yx,
            precomputed_gaussian)

    return image_with_spots


def _add_spot_3d(image, ground_truth, voxel_size_z, voxel_size_yx,
                 precomputed_gaussian):
    """Add a 3-d spot in an image.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 3-d image with shape (z, y, x).
    ground_truth : np.ndarray
        Ground truth array with shape (nb_spots, 6).
        - coordinate_z
        - coordinate_y
        - coordinate_x
        - sigma_z
        - sigma_yx
        - amplitude
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    precomputed_gaussian : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    new_image : np.ndarray, np.uint
        A 3-d image with simulated spots and shape (z, y, x).

    """
    # reshape and cast image
    expectations_raw = np.reshape(image, image.size)
    expectations_raw = expectations_raw.astype(np.float64)

    # build a grid to represent this image
    grid = _initialize_grid_3d(image, voxel_size_z, voxel_size_yx)

    # add spots
    for (coord_z, coord_y, coord_x, sigma_z, sigma_yx, amp) in ground_truth:
        position_spot = np.asarray((coord_z, coord_y, coord_x), dtype=np.int64)
        position_spot = np.ravel_multi_index(position_spot, dims=image.shape)
        position_spot = list(grid[:, position_spot])
        expectations_raw += _gaussian_3d(
            grid=grid,
            mu_z=position_spot[0],
            mu_y=position_spot[1],
            mu_x=position_spot[2],
            sigma_z=sigma_z,
            sigma_yx=sigma_yx,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            psf_amplitude=amp,
            psf_background=0,
            precomputed=precomputed_gaussian)

    # sample Poisson distribution from gaussian values
    image_raw = np.random.poisson(lam=expectations_raw,
                                  size=expectations_raw.size)

    # reshape and cast image
    new_image = np.reshape(image_raw, image.shape)
    new_image = np.clip(new_image, 0, np.iinfo(image.dtype).max)
    new_image = new_image.astype(image.dtype)

    return new_image


def _initialize_grid_3d(image_spot, voxel_size_z, voxel_size_yx):
    """Build a grid in nanometer to compute gaussian function values over a
    full volume.

    Parameters
    ----------
    image_spot : np.ndarray
        A 3-d image with detected spot and shape (z, y, x).
    voxel_size_z : int or float
        Height of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.

    Returns
    -------
    grid : np.ndarray, np.int64
        A grid with the shape (3, z * y * x), in nanometer.

    """
    # get targeted size
    nb_z, nb_y, nb_x = image_spot.shape
    nb_pixels = image_spot.size

    # build meshgrid
    zz, yy, xx = np.meshgrid(np.arange(nb_z), np.arange(nb_y), np.arange(nb_x),
                             indexing="ij")
    zz = zz.astype(np.float32) * float(voxel_size_z)
    yy = yy.astype(np.float32) * float(voxel_size_yx)
    xx = xx.astype(np.float32) * float(voxel_size_yx)

    # format result
    grid = np.zeros((3, nb_pixels), dtype=np.float32)
    grid[0] = np.reshape(zz, (1, nb_pixels))
    grid[1] = np.reshape(yy, (1, nb_pixels))
    grid[2] = np.reshape(xx, (1, nb_pixels))
    grid = np.round(grid).astype(np.int64)

    return grid


def _add_spot_2d(image, ground_truth, voxel_size_yx, precomputed_gaussian):
    """Add a 2-d spot in an image.

    Parameters
    ----------
    image : np.ndarray, np.uint
        A 2-d image with shape (y, x).
    ground_truth : np.ndarray
        Ground truth array with shape (nb_spots, 4).
        - coordinate_y
        - coordinate_x
        - sigma_yx
        - amplitude
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    precomputed_gaussian : Tuple[np.ndarray]
        Tuple with one tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    new_image : np.ndarray, np.uint
        A 2-d image with simulated spots and shape (y, x).

    """
    # reshape and cast image
    expectations_raw = np.reshape(image, image.size)
    expectations_raw = expectations_raw.astype(np.float64)

    # build a grid to represent this image
    grid = _initialize_grid_2d(image, voxel_size_yx)

    # add spots
    for (coord_y, coord_x, sigma_yx, amp) in ground_truth:
        position_spot = np.asarray((coord_y, coord_x), dtype=np.int64)
        position_spot = np.ravel_multi_index(position_spot, dims=image.shape)
        position_spot = list(grid[:, position_spot])
        expectations_raw += _gaussian_2d(
            grid=grid,
            mu_y=position_spot[0],
            mu_x=position_spot[1],
            sigma_yx=sigma_yx,
            voxel_size_yx=voxel_size_yx,
            psf_amplitude=amp,
            psf_background=0,
            precomputed=precomputed_gaussian)

    # sample Poisson distribution from gaussian values
    image_raw = np.random.poisson(lam=expectations_raw,
                                  size=expectations_raw.size)

    # reshape and cast image
    new_image = np.reshape(image_raw, image.shape)
    new_image = np.clip(new_image, 0, np.iinfo(image.dtype).max)
    new_image = new_image.astype(image.dtype)

    return new_image


def _initialize_grid_2d(image_spot, voxel_size_yx):
    """Build a grid in nanometer to compute gaussian function values over a
    full surface.

    Parameters
    ----------
    image_spot : np.ndarray
        A 2-d image with detected spot and shape (y, x).
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.

    Returns
    -------
    grid : np.ndarray, np.int64
        A grid with the shape (2, y * x), in nanometer.

    """
    # get targeted size
    nb_y, nb_x = image_spot.shape
    nb_pixels = image_spot.size

    # build meshgrid
    yy, xx = np.meshgrid(np.arange(nb_y), np.arange(nb_x), indexing="ij")
    yy = yy.astype(np.float32) * float(voxel_size_yx)
    xx = xx.astype(np.float32) * float(voxel_size_yx)

    # format result
    grid = np.zeros((2, nb_pixels), dtype=np.float32)
    grid[0] = np.reshape(yy, (1, nb_pixels))
    grid[1] = np.reshape(xx, (1, nb_pixels))
    grid = np.round(grid).astype(np.int64)

    return grid
