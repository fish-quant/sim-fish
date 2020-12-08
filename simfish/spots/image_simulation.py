# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate smFISH images.
"""

import numpy as np

from skimage.transform import downscale_local_mean

import simfish.utils as utils

from .pattern_simulation import simulate_ground_truth
from .spot_simulation import precompute_erf
from .spot_simulation import add_spots
from .noise_simulation import add_white_noise

# TODO add cluster simulation


def simulate_spots(image_shape=(128, 128), image_dtype=np.uint16,
                   voxel_size_z=None, voxel_size_yx=100,
                   n=30, random_n=False,
                   sigma_z=None, sigma_yx=150, random_sigma=False,
                   amplitude=5000, random_amplitude=False,
                   subpixel_factors=None,
                   noise_level=300,
                   random_level=0.05):
    """Simulate ground truth coordinates and image of spots.

    Parameters
    ----------
    image_shape : Tuple[int or float] or List[int of float]
        Shape (z, y, x) or (y, x) of the image to simulate.
    image_dtype : type
        Type of the image to simulate (np.uint8 or np.uint16).
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
    subpixel_factors : Tuple[int] or List[int]
        Scaling factors to simulate an image with subpixel accuracy. First a
        larger image is simulated, with larger spots, then we downscale it. One
        element per dimension. If None, spots are localized at pixel level.
    noise_level : int or float
        Reference level of noise background to add in the image.
    random_level : float
        Margin allowed to change gaussian parameters provided. The formula
        used is margin = parameter * random_level

    Returns
    -------
    image : np.ndarray, np.uint
        Simulated images with spots and shape (z, y, x) or (y, x).
    ground_truth : np.ndarray
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4).
        - coordinate_z (optional)
        - coordinate_y
        - coordinate_x
        - sigma_z (optional)
        - sigma_yx
        - amplitude

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
    if subpixel_factors is not None:
        if len(subpixel_factors) != ndim:
            raise ValueError("'subpixel_factors' should have {0} elements, "
                             "not {1}.".format(ndim, len(subpixel_factors)))

    # scale image simulation in order to reach subpixel accuracy
    if subpixel_factors is not None:
        image_shape = tuple([image_shape[i] * subpixel_factors[i]
                             for i in range(len(image_shape))])
        if ndim == 3:
            voxel_size_z /= subpixel_factors[0]
        voxel_size_yx /= subpixel_factors[-1]

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

    # adapt image resolution in case of subpixel simulation
    if subpixel_factors is not None:
        image, ground_truth = downscale_image(
            image=image, ground_truth=ground_truth, factors=subpixel_factors)

    # add background noise
    image = add_white_noise(
        image, noise_level=noise_level, random_level=random_level)

    return image, ground_truth


def downscale_image(image, ground_truth, factors):
    """Downscale image and adapt ground truth coordinates.

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
    factors : Tuple[int] or List[int]
        Downscaling factors. One element per dimension.

    Returns
    -------
    image_downscaled : np.ndarray, np.uint
        Image with shape (z/factors, y/factors, x/factors) or
        (y/factors, x/factors).
    ground_truth : np.ndarray
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4).
        Coordinates are downscaled.
        - coordinate_z (optional)
        - coordinate_y
        - coordinate_x
        - sigma_z (optional)
        - sigma_yx
        - amplitude

    """
    # check parameters
    utils.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16])
    utils.check_parameter(factors=(tuple, list))

    # check dimensions
    ndim = len(image.shape)
    if len(factors) != ndim:
        raise ValueError("'factors' should have {0} elements, not {1}."
                         .format(ndim, len(factors)))
    if image.shape[0] % factors[0] != 0:
        raise ValueError("'image' shape is not divisible by 'factors'.")
    if image.shape[1] % factors[1] != 0:
        raise ValueError("'image' shape is not divisible by 'factors'.")
    if ndim == 3 and image.shape[2] % factors[2] != 0:
        raise ValueError("'image' shape is not divisible by 'factors'.")

    # downscale image
    image_downscaled = downscale_local_mean(image, factors=factors,
                                            cval=image.min(), clip=True)

    # adapt coordinates
    # TODO ground truth does not match exactly with image (0.5 drift)
    ground_truth[:, 0] /= factors[0]
    ground_truth[:, 1] /= factors[1]
    if ndim == 3:
        ground_truth[:, 2] /= factors[2]

    # cast image in np.uint
    image_downscaled = np.round(image_downscaled).astype(image.dtype)

    return image_downscaled, ground_truth
