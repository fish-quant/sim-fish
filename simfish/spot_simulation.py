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

def add_spots(image, ground_truth, voxel_size, precomputed_gaussian=None):
    """Simulate gaussian spots based on the ground truth coordinates.

    Parameters
    ----------
    image : np.ndarray
        Image with shape (z, y, x) or (y, x).
    ground_truth : np.ndarray, np.float64
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4). Columns
        are:

        * Coordinate along the z axis (optional).
        * Coordinate along the y axis.
        * Coordinate along the x axis.
        * Standard deviation of the spot along the z axis (optional).
        * Standard deviation of the spot in the yx plan.
        * Intensity of the spot.
    voxel_size : int or float or tuple or list
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). If it's a scalar, the same value is applied to every
        dimensions.
    precomputed_gaussian : sequence of array_like, optional
        Sequence with tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    image_with_spots : np.ndarray
        A 3-d or 2-d image with simulated spots and shape (z, y, x) or (y, x).

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
    stack.check_array(ground_truth, ndim=2, dtype=np.float64)
    stack.check_parameter(voxel_size=(int, float, tuple, list))

    # check consistency between parameters
    ndim = image.ndim
    if isinstance(voxel_size, (tuple, list)):
        if len(voxel_size) != ndim:
            raise ValueError(
                "'voxel_size' must be a scalar or a sequence with {0} "
                "elements.".format(ndim))
    else:
        voxel_size = (voxel_size,) * ndim
    if ndim == 3 and ground_truth.shape[1] != 6:
        raise ValueError("Provided image has 3 dimensions so 'ground_truth' "
                         "should have 6 columns, not {0}."
                         .format(ground_truth.shape[1]))

    # simulate and add 3-d spots...
    if image.ndim == 3:
        image_with_spots = _add_spot_3d(
            image=image,
            ground_truth=ground_truth,
            voxel_size_z=voxel_size[0],
            voxel_size_yx=voxel_size[-1],
            precomputed_gaussian=precomputed_gaussian)

    # ... or 2-d spots
    else:
        image_with_spots = _add_spot_2d(
            image=image,
            ground_truth=ground_truth,
            voxel_size_yx=voxel_size[-1],
            precomputed_gaussian=precomputed_gaussian)

    return image_with_spots


def _add_spot_3d(
        image,
        ground_truth,
        voxel_size_z,
        voxel_size_yx,
        precomputed_gaussian):
    """Add a 3-d gaussian spot in an image.

    Parameters
    ----------
    image : np.ndarray
        A 3-d image with shape (z, y, x).
    ground_truth : np.ndarray, np.float64
        Ground truth array with shape (nb_spots, 6). Columns are:

        * Coordinate along the z axis.
        * Coordinate along the y axis.
        * Coordinate along the x axis.
        * Standard deviation of the spot along the z axis.
        * Standard deviation of the spot in the yx plan.
        * Intensity of the spot.
    voxel_size_z : int or float
        Size of a voxel, along the z axis, in nanometer.
    voxel_size_yx : int or float
        Size of a voxel in the yx plan, in nanometer.
    precomputed_gaussian : sequence of array_like
        Sequence with tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    new_image : np.ndarray
        A 3-d image with simulated spots and shape (z, y, x).

    """
    # cast image
    original_dtype = image.dtype
    if np.issubdtype(original_dtype, np.integer):
        original_max_bound = np.iinfo(original_dtype).max
    else:
        original_max_bound = np.finfo(original_dtype).max
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

    # initialize an empty image and pad it
    image_padded = np.zeros_like(image)
    image_padded = np.pad(
        image_padded,
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

    # sample Poisson distribution for each pixel
    image_padded_raw = np.reshape(image_padded, -1)
    image_padded_raw = np.random.poisson(
        lam=image_padded_raw, size=image_padded_raw.size)
    image_padded = np.reshape(image_padded_raw, image_padded.shape)

    # unpad image
    new_image = image_padded[radius_z:-radius_z,
                             radius_yx:-radius_yx,
                             radius_yx:-radius_yx]

    # cast image
    new_image = np.clip(new_image, 0, original_max_bound)
    new_image = new_image.astype(original_dtype)

    return new_image


def _add_spot_2d(image, ground_truth, voxel_size_yx, precomputed_gaussian):
    """Add a 2-d gaussian spot in an image.

    Parameters
    ----------
    image : np.ndarray
        A 2-d image with shape (y, x).
    ground_truth : np.ndarray, np.float64
        Ground truth array with shape (nb_spots, 4). Columns are:

        * Coordinate along the y axis.
        * Coordinate along the x axis.
        * Standard deviation of the spot in the yx plan.
        * Intensity of the spot.
    voxel_size_yx : int or float
        Size of a voxel in the yx plan, in nanometer.
    precomputed_gaussian : sequence of array_like
        Sequence with tables of precomputed values for the erf, with shape
        (nb_value, 2). One table per dimension.

    Returns
    -------
    new_image : np.ndarray
        A 2-d image with simulated spots and shape (y, x).

    """
    # cast image
    original_dtype = image.dtype
    if np.issubdtype(original_dtype, np.integer):
        original_max_bound = np.iinfo(original_dtype).max
    else:
        original_max_bound = np.finfo(original_dtype).max
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

    # initialize an empty image and pad it
    image_padded = np.zeros_like(image)
    image_padded = np.pad(
        image_padded,
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

    # sample Poisson distribution for each pixel
    image_padded_raw = np.reshape(image_padded, -1)
    image_padded_raw = np.random.poisson(
        lam=image_padded_raw, size=image_padded_raw.size)
    image_padded = np.reshape(image_padded_raw, image_padded.shape)

    # unpad image
    new_image = image_padded[radius_yx:-radius_yx, radius_yx:-radius_yx]

    # cast image
    new_image = np.clip(new_image, 0, original_max_bound)
    new_image = new_image.astype(original_dtype)

    return new_image
