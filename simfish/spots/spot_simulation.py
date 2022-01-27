# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate RNA spots with gaussian functions.
"""

import numpy as np
import bigfish.stack as stack
import bigfish.detection as detection


# ### Gaussian simulation ###

def add_spots(image, ground_truth, voxel_size_z=None, voxel_size_yx=100,
              precomputed_gaussian=None):
    """Simulate gaussian spots based on the ground truth coordinates.

    Parameters
    ----------
    image : np.ndarray, np.uint
        Image with shape (z, y, x) or (y, x).
    ground_truth : np.ndarray, np.float64
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
    stack.check_array(image, ndim=[2, 3], dtype=[np.uint8, np.uint16])
    stack.check_array(ground_truth, ndim=2, dtype=np.float64)
    stack.check_parameter(
        voxel_size_z=(int, float, type(None)),
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
            image=image,
            ground_truth=ground_truth,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            precomputed_gaussian=precomputed_gaussian)

    # ... or 2-d spots
    else:
        image_with_spots = _add_spot_2d(
            image=image,
            ground_truth=ground_truth,
            voxel_size_yx=voxel_size_yx,
            precomputed_gaussian=precomputed_gaussian)

    return image_with_spots


def _add_spot_3d(image, ground_truth, voxel_size_z, voxel_size_yx,
                 precomputed_gaussian):
    """Add a 3-d gaussian spot in an image.

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
    # cast image
    original_dtype = image.dtype
    image = image.astype(np.float64)

    # compute reference spot shape
    max_sigma_z = max(ground_truth[:, 3])
    max_sigma_yx = max(ground_truth[:, 4])
    radius_pixel = detection.get_object_radius_pixel(
        voxel_size_nm=(voxel_size_z, voxel_size_yx, voxel_size_yx),
        object_radius_nm=(max_sigma_z, max_sigma_yx, max_sigma_yx),
        ndim=3)
    radius = [np.sqrt(3) * r for r in radius_pixel]
    radius_z = np.ceil(radius[0]).astype(np.int64)
    z_shape = radius_z * 2 + 1
    radius_yx = np.ceil(radius[-1]).astype(np.int64)
    yx_shape = radius_yx * 2 + 1

    # build a grid to represent a spot image
    image_spot = np.zeros((z_shape, yx_shape, yx_shape), dtype=np.uint8)
    grid = detection.initialize_grid(
        image_spot=image_spot,
        voxel_size=(voxel_size_z, voxel_size_yx, voxel_size_yx),
        return_centroid=False)

    # pad image
    image_padded = np.pad(
        image,
        pad_width=((radius_z, radius_z),
                   (radius_yx, radius_yx),
                   (radius_yx, radius_yx)),
        mode="constant")

    # loop over every spot
    for (coord_z, coord_y, coord_x, sigma_z, sigma_yx, amp) in ground_truth:

        # simulate spot signal
        position_spot = np.asarray((radius_z, radius_yx, radius_yx),
                                   dtype=np.int64)
        position_spot = np.ravel_multi_index(
            position_spot, dims=image_spot.shape)
        position_spot = list(grid[:, position_spot])
        simulated_spot = detection.gaussian_3d(
            grid=grid,
            mu_z=position_spot[0],
            mu_y=position_spot[1],
            mu_x=position_spot[2],
            sigma_z=sigma_z,
            sigma_yx=sigma_yx,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            amplitude=amp,
            background=0,
            precomputed=precomputed_gaussian)
        simulated_spot = np.reshape(simulated_spot, image_spot.shape)

        # add spot
        coord_z_min = int(coord_z)
        coord_z_max = int(coord_z + 2 * radius_z + 1)
        coord_y_min = int(coord_y)
        coord_y_max = int(coord_y + 2 * radius_yx + 1)
        coord_x_min = int(coord_x)
        coord_x_max = int(coord_x + 2 * radius_yx + 1)
        image_padded[coord_z_min:coord_z_max,
                     coord_y_min:coord_y_max,
                     coord_x_min:coord_x_max] += simulated_spot

    # unpad image
    image = image_padded[radius_z:-radius_z,
                         radius_yx:-radius_yx,
                         radius_yx:-radius_yx]
    image_raw = np.reshape(image, image.size)

    # sample Poisson distribution from gaussian values
    image_raw = np.random.poisson(lam=image_raw, size=image_raw.size)

    # reshape and cast image
    new_image = np.reshape(image_raw, image.shape)
    new_image = np.clip(new_image, 0, np.iinfo(original_dtype).max)
    new_image = new_image.astype(original_dtype)

    return new_image


def _add_spot_3d_bis(image, ground_truth, voxel_size_z, voxel_size_yx,
                     precomputed_gaussian):
    """Add a 3-d gaussian spot in an image.

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
    grid = detection.initialize_grid(
        image_spot=image,
        voxel_size=(voxel_size_z, voxel_size_yx, voxel_size_yx),
        return_centroid=False)

    # add spots
    for (coord_z, coord_y, coord_x, sigma_z, sigma_yx, amp) in ground_truth:
        position_spot = np.asarray((coord_z, coord_y, coord_x), dtype=np.int64)
        position_spot = np.ravel_multi_index(position_spot, dims=image.shape)
        position_spot = list(grid[:, position_spot])
        expectations_raw += detection.gaussian_3d(
            grid=grid,
            mu_z=position_spot[0],
            mu_y=position_spot[1],
            mu_x=position_spot[2],
            sigma_z=sigma_z,
            sigma_yx=sigma_yx,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            amplitude=amp,
            background=0,
            precomputed=precomputed_gaussian)

    # sample Poisson distribution from gaussian values
    image_raw = np.random.poisson(
        lam=expectations_raw, size=expectations_raw.size)

    # reshape and cast image
    new_image = np.reshape(image_raw, image.shape)
    new_image = np.clip(new_image, 0, np.iinfo(image.dtype).max)
    new_image = new_image.astype(image.dtype)

    return new_image


def _add_spot_2d(image, ground_truth, voxel_size_yx, precomputed_gaussian):
    """Add a 2-d gaussian spot in an image.

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
    # cast image
    original_dtype = image.dtype
    image = image.astype(np.float64)

    # compute reference spot shape
    max_sigma = max(ground_truth[:, 2])
    radius_pixel = detection.get_object_radius_pixel(
        voxel_size_nm=(voxel_size_yx, voxel_size_yx),
        object_radius_nm=(max_sigma, max_sigma),
        ndim=2)
    radius = [np.sqrt(3) * r for r in radius_pixel]
    radius_yx = np.ceil(radius[-1]).astype(np.int64)
    yx_shape = radius_yx * 2 + 1

    # build a grid to represent a spot image
    image_spot = np.zeros((yx_shape, yx_shape), dtype=np.uint8)
    grid = detection.initialize_grid(
        image_spot=image_spot,
        voxel_size=(voxel_size_yx, voxel_size_yx),
        return_centroid=False)

    # pad image
    image_padded = np.pad(
        image,
        pad_width=((radius_yx, radius_yx),
                   (radius_yx, radius_yx)),
        mode="constant")

    # loop over every spot
    for (coord_y, coord_x, sigma_yx, amp) in ground_truth:

        # simulate spot signal
        position_spot = np.asarray((radius_yx, radius_yx), dtype=np.int64)
        position_spot = np.ravel_multi_index(
            position_spot, dims=image_spot.shape)
        position_spot = list(grid[:, position_spot])
        simulated_spot = detection.gaussian_2d(
            grid=grid,
            mu_y=position_spot[0],
            mu_x=position_spot[1],
            sigma_yx=sigma_yx,
            voxel_size_yx=voxel_size_yx,
            amplitude=amp,
            background=0,
            precomputed=precomputed_gaussian)
        simulated_spot = np.reshape(simulated_spot, image_spot.shape)

        # add spot
        coord_y_min = int(coord_y)
        coord_y_max = int(coord_y + 2 * radius_yx + 1)
        coord_x_min = int(coord_x)
        coord_x_max = int(coord_x + 2 * radius_yx + 1)
        image_padded[coord_y_min:coord_y_max,
                     coord_x_min:coord_x_max] += simulated_spot

    # unpad image
    image = image_padded[radius_yx:-radius_yx, radius_yx:-radius_yx]
    image_raw = np.reshape(image, image.size)

    # sample Poisson distribution from gaussian values
    image_raw = np.random.poisson(lam=image_raw, size=image_raw.size)

    # reshape and cast image
    new_image = np.reshape(image_raw, image.shape)
    new_image = np.clip(new_image, 0, np.iinfo(original_dtype).max)
    new_image = new_image.astype(original_dtype)

    return new_image
