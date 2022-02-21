# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to simulate smFISH images.
"""

import numpy as np
import bigfish.stack as stack
import bigfish.detection as detection

from .pattern_simulation import simulate_ground_truth
from .spot_simulation import add_spots
from .noise_simulation import add_white_noise

from skimage.transform import downscale_local_mean


def simulate_images(
        n_images,
        ndim,
        n_spots,
        random_n_spots=False,
        n_clusters=0,
        random_n_clusters=False,
        n_spots_cluster=0,
        random_n_spots_cluster=False,
        centered_cluster=False,
        image_shape=(128, 128),
        image_dtype=np.uint16,
        subpixel_factors=None,
        voxel_size=(100, 100),
        sigma=(150, 150),
        random_sigma=0.05,
        amplitude=5000,
        random_amplitude=0.05,
        noise_level=300,
        random_noise=0.05):
    """Return a generator to simulate multiple ground truth coordinates and
    images of spots.

    Parameters
    ----------
    n_images : int
        Number of images to simulate.
    ndim : {2, 3}
        Number of dimension to consider for the simulation.
    n_spots : int or tuple
        Expected number of spots to simulate per image. If tuple, provide the
        minimum and maximum number of spots to simulate. Multiple images are
        simulated with a growing number of spots.
    random_n_spots : bool, default=False
        Make the number of spots follow a Poisson distribution with
        expectation `n_spots`, instead of a constant predefined value.
    n_clusters : int
        Expected number of clusters to simulate.
    random_n_clusters : bool, default=False
        Make the number of spots follow a Poisson distribution with
        expectation `n_clusters`, instead of a constant predefined value.
    n_spots_cluster : int
        Expected number of spots per cluster to simulate.
    random_n_spots_cluster : bool, default=False
        Make the number of spots follow a Poisson distribution with
        expectation `n_spots_cluster`, instead of a constant predefined value.
    centered_cluster : bool, default=False
        Center the simulated cluster. Only used one cluster is simulated.
    image_shape : tuple or list, default=(128, 128)
        Shape (z, y, x) or (y, x) of the images to simulate.
    image_dtype : dtype, default=np.uint16
        Type of the image to simulate (np.uint8, np.uint16, np.float32 or
        np.float64).
    subpixel_factors : tuple or list, optional
        Scaling factors to simulate images with subpixel accuracy. First a
        larger image is simulated, with larger spots, then we downscale it. One
        element per dimension. Can make the simulation much longer. If None,
        spots are localized at pixel level.
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
    noise_level : int or float, default=300
        Reference level of noise background to add in the image.
    random_noise : int or float
        Background noise follows a normal distribution around the provided
        noise values. The scale used is:

        .. math::
            \\mbox{scale} = \\mbox{noise_level} * \\mbox{random_noise}

    Returns
    -------
    _ : Tuple generator
        image : np.ndarray
            Simulated image with spots and shape (z, y, x) or (y, x).
        ground_truth : np.ndarray, np.float64
            Ground truth array with shape (nb_spots, 6) or (nb_spots, 4).
            Columns are:

            * Coordinate along the z axis (optional).
            * Coordinate along the y axis.
            * Coordinate along the x axis.
            * Standard deviation of the spot along the z axis (optional).
            * Standard deviation of the spot in the yx plan.
            * Intensity of the spot.

    """
    # check parameters
    stack.check_parameter(
        n_images=int,
        n_spots=(int, tuple))

    # check number of images
    if n_images <= 0:
        raise ValueError("'n_images' should be strictly positive.")

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
            ndim=ndim,
            n_spots=n_spots,
            random_n_spots=random_n_spots,
            n_clusters=n_clusters,
            random_n_clusters=random_n_clusters,
            n_spots_cluster=n_spots_cluster,
            random_n_spots_cluster=random_n_spots_cluster,
            centered_cluster=centered_cluster,
            image_shape=image_shape,
            image_dtype=image_dtype,
            subpixel_factors=subpixel_factors,
            voxel_size=voxel_size,
            sigma=sigma,
            random_sigma=random_sigma,
            amplitude=amplitude,
            random_amplitude=random_amplitude,
            noise_level=noise_level,
            random_noise=random_noise)

        yield image, ground_truth


def simulate_image(
        ndim,
        n_spots,
        random_n_spots=False,
        n_clusters=0,
        random_n_clusters=False,
        n_spots_cluster=0,
        random_n_spots_cluster=False,
        centered_cluster=False,
        image_shape=(128, 128),
        image_dtype=np.uint16,
        subpixel_factors=None,
        voxel_size=(100, 100),
        sigma=(150, 150),
        random_sigma=0.05,
        amplitude=5000,
        random_amplitude=0.05,
        noise_level=300,
        random_noise=0.05):
    """Simulate ground truth coordinates and image of spots.

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
        Make the number of spots follow a Poisson distribution with
        expectation `n_clusters`, instead of a constant predefined value.
    n_spots_cluster : int
        Expected number of spots per cluster to simulate.
    random_n_spots_cluster : bool, default=False
        Make the number of spots follow a Poisson distribution with
        expectation `n_spots_cluster`, instead of a constant predefined value.
    centered_cluster : bool, default=False
        Center the simulated cluster. Only used one cluster is simulated.
    image_shape : tuple or list, default=(128, 128)
        Shape (z, y, x) or (y, x) of the image to simulate.
    image_dtype : dtype, default=np.uint16
        Type of the image to simulate (np.uint8, np.uint16, np.float32 or
        np.float64).
    subpixel_factors : tuple or list, optional
        Scaling factors to simulate an image with subpixel accuracy. First a
        larger image is simulated, with larger spots, then we downscale it. One
        element per dimension. Can make the simulation much longer. If None,
        spots are localized at pixel level.
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
    noise_level : int or float, default=300
        Reference level of noise background to add in the image.
    random_noise : int or float
        Background noise follows a normal distribution around the provided
        noise values. The scale used is:

        .. math::
            \\mbox{scale} = \\mbox{noise_level} * \\mbox{random_noise}

    Returns
    -------
    image : np.ndarray, np.uint
        Simulated image with spots and shape (z, y, x) or (y, x).
    ground_truth : np.ndarray
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4):

        * `coordinate_z` (optional)
        * `coordinate_y`
        * `coordinate_x`
        * `sigma_z` (optional)
        * `sigma_yx`
        * `amplitude`

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
        image_shape=(tuple, list),
        image_dtype=type,
        subpixel_factors=(tuple, list, type(None)),
        voxel_size=(int, float, tuple, list),
        sigma=(int, float, tuple, list),
        random_sigma=(int, float),
        amplitude=(int, float),
        random_amplitude=(int, float),
        noise_level=(int, float),
        random_noise=(int, float))

    # check image dtype
    if image_dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
        raise ValueError("'image_dtype' should be np.uint8, np.uint16, "
                         "np.float32 or np.float64 not {0}."
                         .format(image_dtype))

    # check consistency between parameters
    if ndim not in [2, 3]:
        raise ValueError("'ndim' should have 2 or 3 elements, not {0}."
                         .format(ndim))
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
    if len(image_shape) != ndim:
        raise ValueError("'image_shape' should have {0} elements, not {0}."
                         .format(ndim, len(image_shape)))
    if subpixel_factors is not None:
        if len(subpixel_factors) != ndim:
            raise ValueError(
                "'subpixel_factors' should have {0} elements, "
                "not {1}.".format(ndim, len(subpixel_factors)))

    # scale image simulation in order to reach subpixel accuracy
    image_shape, voxel_size = _scale_subpixel(
        ndim=ndim,
        image_shape=image_shape,
        subpixel_factors=subpixel_factors,
        voxel_size=voxel_size)

    # initialize image
    image = np.zeros(image_shape, dtype=image_dtype)

    # generate ground truth
    ground_truth = simulate_ground_truth(
        ndim=ndim,
        n_spots=n_spots,
        random_n_spots=random_n_spots,
        n_clusters=n_clusters,
        random_n_clusters=random_n_clusters,
        n_spots_cluster=n_spots_cluster,
        random_n_spots_cluster=random_n_spots_cluster,
        centered_cluster=centered_cluster,
        frame_shape=image_shape,
        voxel_size=voxel_size,
        sigma=sigma,
        random_sigma=random_sigma,
        amplitude=amplitude,
        random_amplitude=random_amplitude)

    # skip these steps if no spots are simulated
    if len(ground_truth) > 0:

        # precompute spots if possible
        precomputed_erf = _precompute_gaussian(
            ground_truth=ground_truth,
            random_sigma=random_sigma,
            voxel_size=voxel_size,
            sigma=sigma)

        # simulate spots
        image = add_spots(
            image=image,
            ground_truth=ground_truth,
            voxel_size=voxel_size,
            precomputed_gaussian=precomputed_erf)

    # adapt image resolution in case of subpixel simulation
    if subpixel_factors is not None:
        image, ground_truth = _downscale_image(
            image=image,
            ground_truth=ground_truth,
            factors=subpixel_factors)

    # add background noise
    image = add_white_noise(
        image=image,
        noise_level=noise_level,
        random_noise=random_noise)

    return image, ground_truth


def _scale_subpixel(ndim, image_shape, subpixel_factors, voxel_size):
    """Upscale the image to perform a subpixel simulation.

    Parameters
    ----------
    ndim : {2, 3}
        Number of dimension to consider for the simulation.
    image_shape : tuple or list
        Shape (z, y, x) or (y, x) of the image to simulate.
    subpixel_factors : tuple or list or None
        Scaling factors to simulate an image with subpixel accuracy.
    voxel_size : tuple or list
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions).

    Returns
    -------
    image_shape : tuple
        Upscale shape (z, y, x) or (y, x) of the image to simulate.
    voxel_size : tuple or list
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions). Size is consistent with the upscaled image.

    """
    # scale image simulation in order to reach subpixel accuracy
    if subpixel_factors is not None:
        image_shape = tuple([image_shape[i] * subpixel_factors[i]
                             for i in range(len(image_shape))])
        voxel_size_yx = voxel_size[-1] / subpixel_factors[-1]
        if ndim == 3:
            voxel_size_z = voxel_size[0] / subpixel_factors[0]
            voxel_size = (voxel_size_z, voxel_size_yx, voxel_size_yx)
        else:
            voxel_size = (voxel_size_yx, voxel_size_yx)

    return image_shape, voxel_size


def _precompute_gaussian(ground_truth, random_sigma, voxel_size, sigma):
    """Precompute different values for the gaussian signal.

    Parameters
    ----------
    ground_truth : np.ndarray, np.float64
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4). Columns
        are:

        * Coordinate along the z axis (optional).
        * Coordinate along the y axis.
        * Coordinate along the x axis.
        * Standard deviation of the spot along the z axis (optional).
        * Standard deviation of the spot in the yx plan.
        * Intensity of the spot.
    random_sigma : int or float
        Sigmas follow a normal distribution around the provided sigma values.
        The scale used is:

        .. math::
            \\mbox{scale} = \\mbox{sigma} * \\mbox{random_sigma}
    voxel_size : tuple or list
        Size of a voxel, in nanometer. One value per spatial dimension (zyx or
        yx dimensions).
    sigma : tuple or list
        Standard deviation of the gaussian, in nanometer. One value per spatial
        dimension (zyx or yx dimensions).

    Returns
    -------
    precomputed_gaussian : sequence of array_like or None
        Sequence with tables of precomputed values for the gaussian signal,
        with shape (nb_value, 2). One table per dimension.

    """
    # precompute gaussian spots if possible
    if random_sigma == 0:

        if ground_truth.shape[1] == 6:
            max_sigma_z = max(ground_truth[:, 3])
            max_sigma_yx = max(ground_truth[:, 4])
            radius_pixel = detection.get_object_radius_pixel(
                voxel_size_nm=voxel_size,
                object_radius_nm=(max_sigma_z, max_sigma_yx, max_sigma_yx),
                ndim=3)
            radius = [np.sqrt(3) * r for r in radius_pixel]
            radius_z = np.ceil(radius[0]).astype(np.int64)
            z_shape = radius_z * 2 + 1
            radius_yx = np.ceil(radius[-1]).astype(np.int64)
            yx_shape = radius_yx * 2 + 1
            max_size = int(max(z_shape, yx_shape) + 1)
            precomputed_gaussian = detection.precompute_erf(
                ndim=3,
                voxel_size=voxel_size,
                sigma=sigma,
                max_grid=max_size)

        else:
            max_sigma_yx = max(ground_truth[:, 2])
            radius_pixel = detection.get_object_radius_pixel(
                voxel_size_nm=voxel_size,
                object_radius_nm=(max_sigma_yx, max_sigma_yx),
                ndim=2)
            radius = [np.sqrt(3) * r for r in radius_pixel]
            radius_yx = np.ceil(radius[-1]).astype(np.int64)
            yx_shape = radius_yx * 2 + 1
            max_size = int(yx_shape + 1)
            precomputed_gaussian = detection.precompute_erf(
                ndim=2,
                voxel_size=voxel_size,
                sigma=sigma,
                max_grid=max_size)

    else:
        precomputed_gaussian = None

    return precomputed_gaussian


def _downscale_image(image, ground_truth, factors):
    """Downscale image and adapt ground truth coordinates.

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
    factors : tuple or list
        Downscaling factors. One element per dimension.

    Returns
    -------
    image_downscaled : np.ndarray
        Image with shape (z/factors, y/factors, x/factors) or
        (y/factors, x/factors).
    ground_truth : np.ndarray, np.float64
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4). Columns
        are:

        * Coordinate downscaled along the z axis (optional).
        * Coordinate downscaled along the y axis.
        * Coordinate downscaled along the x axis.
        * Standard deviation of the spot along the z axis (optional).
        * Standard deviation of the spot in the yx plan.
        * Intensity of the spot.

    """
    # check parameters
    stack.check_array(
        image,
        ndim=[2, 3],
        dtype=[np.uint8, np.uint16, np.float32, np.float64])
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
    image_downscaled = downscale_local_mean(
        image, factors=factors, cval=image.min(), clip=True)

    # adapt coordinates
    ground_truth[:, 0] /= factors[0]
    ground_truth[:, 1] /= factors[1]
    if ndim == 3:
        ground_truth[:, 2] /= factors[2]

    # cast image in np.uint
    image_downscaled = np.round(image_downscaled).astype(image.dtype)

    return image_downscaled, ground_truth
