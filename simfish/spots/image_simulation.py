# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate smFISH images.
"""

import numpy as np
import bigfish.stack as stack
import bigfish.detection as detection

from skimage.transform import downscale_local_mean


from .pattern_simulation import simulate_ground_truth
from .spot_simulation import add_spots
from .noise_simulation import add_white_noise


def simulate_images(n_images, image_shape=(128, 128), image_dtype=np.uint16,
                    subpixel_factors=None,
                    voxel_size_z=None, voxel_size_yx=100,
                    n_spots=30, random_n_spots=False,
                    n_clusters=0, random_n_clusters=False,
                    n_spots_cluster=0, random_n_spots_cluster=False,
                    centered_cluster=False,
                    sigma_z=None, sigma_yx=150, random_sigma=0,
                    amplitude=5000, random_amplitude=0.05,
                    noise_level=300, random_noise=0.05):
    """Simulate ground truth coordinates and images of spots.

    Parameters
    ----------
    n_images : int
        Number of images to simulate.
    image_shape : Tuple[int or float] or List[int of float]
        Shape (z, y, x) or (y, x) of the image to simulate.
    image_dtype : type
        Type of the image to simulate (np.uint8 or np.uint16).
    subpixel_factors : Tuple[int] or List[int]
        Scaling factors to simulate an image with subpixel accuracy. First a
        larger image is simulated, with larger spots, then we downscale it. One
        element per dimension. If None, spots are localized at pixel level.
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we
        consider a 2-d image.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    n_spots : int or Tuple[int]
        Expected number of spots to simulate per image. If tuple, provide the
        minimum and maximum number of spots to simulate. Multiple images are
        simulated with a growing number of spots.
    random_n_spots : bool
        Make the number of spots follow a Poisson distribution with
        expectation n_spots, instead of a constant predefined value.
    n_clusters : int
        Expected number of clusters to simulate.
    random_n_clusters : bool
        Make the number of spots follow a Poisson distribution with
        expectation n_clusters, instead of a constant predefined value.
    n_spots_cluster : int
        Expected number of spots per cluster to simulate.
    random_n_spots_cluster : bool
        Make the number of spots follow a Poisson distribution with
        expectation n_spots_cluster, instead of a constant predefined value.
    centered_cluster : bool
        Center the simulated cluster. Only used one cluster is simulated.
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
    noise_level : int or float
        Reference level of noise background to add in the image.
    random_noise : int or float
        Background noise follows a normal distribution around the provided
        noise values. The scale used is scale = noise_level * random_noise

    Returns
    -------
    _ : Tuple generator
        image : np.ndarray, np.uint
            Simulated image with spots and shape (z, y, x) or (y, x).
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
    stack.check_parameter(n_images=int,
                          image_shape=(tuple, list),
                          image_dtype=type,
                          subpixel_factors=(tuple, type(None)),
                          voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          n_spots=(int, tuple),
                          random_n_spots=bool,
                          n_clusters=int,
                          random_n_clusters=bool,
                          n_spots_cluster=int,
                          random_n_spots_cluster=bool,
                          centered_cluster=bool,
                          sigma_z=(int, float, type(None)),
                          sigma_yx=(int, float),
                          random_sigma=(int, float),
                          amplitude=(int, float),
                          random_amplitude=(int, float),
                          noise_level=(int, float),
                          random_noise=(int, float))

    # check number of images
    if n_images < 0:
        raise ValueError("'n_images' should be positive.")

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

    # define number of spots
    if isinstance(n_spots, tuple):
        l_n = np.linspace(n_spots[0], n_spots[1], num=n_images, dtype=np.int64)
    else:
        l_n = None

    # simulate images
    for i in range(n_images):
        if l_n is not None:
            n_spots = int(l_n[i])
        image, ground_truth = simulate_image(
            image_shape=image_shape,
            image_dtype=image_dtype,
            subpixel_factors=subpixel_factors,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            n_spots=n_spots,
            random_n_spots=random_n_spots,
            n_clusters=n_clusters,
            random_n_clusters=random_n_clusters,
            n_spots_cluster=n_spots_cluster,
            random_n_spots_cluster=random_n_spots_cluster,
            centered_cluster=centered_cluster,
            sigma_z=sigma_z,
            sigma_yx=sigma_yx,
            random_sigma=random_sigma,
            amplitude=amplitude,
            random_amplitude=random_amplitude,
            noise_level=noise_level,
            random_noise=random_noise)

        yield image, ground_truth


def simulate_image(image_shape=(128, 128), image_dtype=np.uint16,
                   subpixel_factors=None,
                   voxel_size_z=None, voxel_size_yx=100,
                   n_spots=30, random_n_spots=False,
                   n_clusters=0, random_n_clusters=False,
                   n_spots_cluster=0, random_n_spots_cluster=False,
                   centered_cluster=False,
                   sigma_z=None, sigma_yx=150, random_sigma=0.05,
                   amplitude=5000, random_amplitude=0.05,
                   noise_level=300, random_noise=0.05):
    """Simulate ground truth coordinates and image of spots.

    Parameters
    ----------
    image_shape : Tuple[int or float] or List[int of float]
        Shape (z, y, x) or (y, x) of the image to simulate.
    image_dtype : type
        Type of the image to simulate (np.uint8 or np.uint16).
    subpixel_factors : Tuple[int] or List[int]
        Scaling factors to simulate an image with subpixel accuracy. First a
        larger image is simulated, with larger spots, then we downscale it. One
        element per dimension. If None, spots are localized at pixel level.
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we
        consider a 2-d image.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    n_spots : int
        Expected number of spots to simulate.
    random_n_spots : bool
        Make the number of spots follow a Poisson distribution with
        expectation n_spots, instead of a constant predefined value.
    n_clusters : int
        Expected number of clusters to simulate.
    random_n_clusters : bool
        Make the number of spots follow a Poisson distribution with
        expectation n_clusters, instead of a constant predefined value.
    n_spots_cluster : int
        Expected number of spots per cluster to simulate.
    random_n_spots_cluster : bool
        Make the number of spots follow a Poisson distribution with
        expectation n_spots_cluster, instead of a constant predefined value.
    centered_cluster : bool
        Center the simulated cluster. Only used one cluster is simulated.
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
    noise_level : int or float
        Reference level of noise background to add in the image.
    random_noise : int or float
        Background noise follows a normal distribution around the provided
        noise values. The scale used is scale = noise_level * random_noise

    Returns
    -------
    image : np.ndarray, np.uint
        Simulated image with spots and shape (z, y, x) or (y, x).
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
    stack.check_parameter(image_shape=(tuple, list),
                          image_dtype=type,
                          subpixel_factors=(tuple, type(None)),
                          voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          n_spots=int,
                          random_n_spots=bool,
                          n_clusters=int,
                          random_n_clusters=bool,
                          n_spots_cluster=int,
                          random_n_spots_cluster=bool,
                          centered_cluster=bool,
                          sigma_z=(int, float, type(None)),
                          sigma_yx=(int, float),
                          random_sigma=(int, float),
                          amplitude=(int, float),
                          random_amplitude=(int, float),
                          noise_level=(int, float),
                          random_noise=(int, float))

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
    image_shape, voxel_size_z, voxel_size_yx = _scale_subpixel(
        image_shape=image_shape,
        subpixel_factors=subpixel_factors,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx)

    # initialize image
    image = np.zeros(image_shape, dtype=image_dtype)

    # compensate noise level in amplitude
    amplitude -= noise_level

    # generate ground truth
    ground_truth = simulate_ground_truth(
        n_spots=n_spots,
        random_n_spots=random_n_spots,
        n_clusters=n_clusters,
        random_n_clusters=random_n_clusters,
        n_spots_cluster=n_spots_cluster,
        random_n_spots_cluster=random_n_spots_cluster,
        centered_cluster=centered_cluster,
        frame_shape=image_shape,
        voxel_size_z=voxel_size_z,
        voxel_size_yx=voxel_size_yx,
        sigma_z=sigma_z,
        sigma_yx=sigma_yx,
        random_sigma=random_sigma,
        amplitude=amplitude,
        random_amplitude=random_amplitude)

    # skip these steps if no spots are simulated
    if len(ground_truth) > 0:

        # precompute spots if possible
        precomputed_erf = _precompute_gaussian(
            ground_truth=ground_truth,
            random_sigma=random_sigma,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            sigma_z=sigma_z,
            sigma_yx=sigma_yx)

        # simulate spots
        image = add_spots(
            image, ground_truth,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            precomputed_gaussian=precomputed_erf)

    # adapt image resolution in case of subpixel simulation
    if subpixel_factors is not None:
        image, ground_truth = downscale_image(
            image=image, ground_truth=ground_truth, factors=subpixel_factors)

    # add background noise
    image = add_white_noise(
        image, noise_level=noise_level, random_noise=random_noise)

    return image, ground_truth


def _scale_subpixel(image_shape, subpixel_factors, voxel_size_z,
                    voxel_size_yx):
    # get number of dimensions
    ndim = len(image_shape)

    # scale image simulation in order to reach subpixel accuracy
    if subpixel_factors is not None:
        image_shape = tuple([image_shape[i] * subpixel_factors[i]
                             for i in range(len(image_shape))])
        if ndim == 3:
            voxel_size_z /= subpixel_factors[0]
        voxel_size_yx /= subpixel_factors[-1]

    return image_shape, voxel_size_z, voxel_size_yx


def _precompute_gaussian(ground_truth, random_sigma, voxel_size_z,
                         voxel_size_yx, sigma_z, sigma_yx):
    # precompute gaussian spots if possible
    if random_sigma == 0:

        if ground_truth.shape[1] == 6:
            max_sigma_z = max(ground_truth[:, 3])
            max_sigma_yx = max(ground_truth[:, 4])
            radius_z, radius_yx, _ = stack.get_radius(
                voxel_size_z=voxel_size_z, voxel_size_yx=voxel_size_yx,
                psf_z=max_sigma_z, psf_yx=max_sigma_yx)
            radius_z = np.ceil(radius_z).astype(np.int64)
            z_shape = radius_z * 2 + 1
            radius_yx = np.ceil(radius_yx).astype(np.int64)
            yx_shape = radius_yx * 2 + 1
            max_size = int(max(z_shape, yx_shape) + 1)
            precomputed_erf = detection.precompute_erf(
                voxel_size_z=voxel_size_z, voxel_size_yx=voxel_size_yx,
                sigma_z=sigma_z, sigma_yx=sigma_yx, max_grid=max_size)

        else:
            max_sigma_yx = max(ground_truth[:, 2])
            radius_yx, _ = stack.get_radius(
                voxel_size_z=None, voxel_size_yx=voxel_size_yx,
                psf_z=None, psf_yx=max_sigma_yx)
            radius_yx = np.ceil(radius_yx).astype(np.int64)
            yx_shape = radius_yx * 2 + 1
            max_size = int(yx_shape + 1)
            precomputed_erf = detection.precompute_erf(
                voxel_size_z=None, voxel_size_yx=voxel_size_yx,
                sigma_z=None, sigma_yx=sigma_yx, max_grid=max_size)

    else:
        precomputed_erf = None

    return precomputed_erf


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
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16])
    stack.check_parameter(factors=(tuple, list))

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
    ground_truth[:, 0] /= factors[0]
    ground_truth[:, 1] /= factors[1]
    if ndim == 3:
        ground_truth[:, 2] /= factors[2]

    # cast image in np.uint
    image_downscaled = np.round(image_downscaled).astype(image.dtype)

    return image_downscaled, ground_truth
