# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for simfish.spots subpackage.
"""

import simfish.utils as utils
import numpy as np


# ### Sigma and radius ###

def get_sigma(voxel_size_z=None, voxel_size_yx=100, psf_z=None, psf_yx=200):
    """Compute the standard deviation of the PSF of the spots.

    Parameters
    ----------
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we
        consider a 2-d PSF.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer. If None, we consider a 2-d PSF.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.

    Returns
    -------
    sigma : Tuple[float]
        Standard deviations in pixel of the PSF, one element per dimension.

    """
    # check parameters
    utils.check_parameter(voxel_size_z=(int, float, type(None)),
                          voxel_size_yx=(int, float),
                          psf_z=(int, float, type(None)),
                          psf_yx=(int, float))

    # compute sigma
    sigma_yx = psf_yx / voxel_size_yx

    if voxel_size_z is None or psf_z is None:
        return sigma_yx, sigma_yx

    else:
        sigma_z = psf_z / voxel_size_z
        return sigma_z, sigma_yx, sigma_yx


def get_radius(voxel_size_z=None, voxel_size_yx=100, psf_z=None, psf_yx=200):
    """Approximate the radius of the detected spot.

    We use the formula:

        sqrt(ndim) * sigma

    with ndim the number of dimension of the image and sigma the standard
    deviation (in pixel) of the detected spot.

    Parameters
    ----------
    voxel_size_z : int or float or None
        Height of a voxel, along the z axis, in nanometer. If None, we
        consider a 2-d spot.
    voxel_size_yx : int or float
        Size of a voxel on the yx plan, in nanometer.
    psf_z : int or float or None
        Theoretical size of the PSF emitted by a spot in the z plan,
        in nanometer. If None, we consider a 2-d spot.
    psf_yx : int or float
        Theoretical size of the PSF emitted by a spot in the yx plan,
        in nanometer.

    Returns
    -------
    radius : Tuple[float]
        Radius in pixels of the detected spots, one element per dimension.

    """
    # compute sigma
    sigma = get_sigma(voxel_size_z, voxel_size_yx, psf_z, psf_yx)

    # compute radius
    radius = [np.sqrt(len(sigma)) * sigma_ for sigma_ in sigma]
    radius = tuple(radius)

    return radius
