# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to plot 2-d pixel and coordinates images.
"""

import simfish.utils as utils

from .utils import save_plot, get_minmax_values

import matplotlib.pyplot as plt
import numpy as np


# ### General plot ###

def plot_yx(image, r=0, c=0, z=0, rescale=False, contrast=False,
            title=None, framesize=(8, 8), remove_frame=True, path_output=None,
            ext="png", show=True):
    """Plot the selected yx plan of the selected dimensions of an image.

    Parameters
    ----------
    image : np.ndarray
        A 2-d, 3-d, 4-d or 5-d image with shape (y, x), (z, y, x),
        (c, z, y, x) or (r, c, z, y, x) respectively.
    r : int
        Index of the round to keep.
    c : int
        Index of the channel to keep.
    z : int
        Index of the z slice to keep.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    title : str
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    remove_frame : bool
        Remove axes and frame.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    Returns
    -------

    """
    # check parameters
    utils.check_array(image,
                      ndim=[2, 3, 4, 5],
                      dtype=[np.uint8, np.uint16, np.int64,
                             np.float32, np.float64,
                             bool])
    utils.check_parameter(r=int, c=int, z=int,
                          rescale=bool,
                          contrast=bool,
                          title=(str, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list))

    # get the 2-d image
    if image.ndim == 2:
        xy_image = image
    elif image.ndim == 3:
        xy_image = image[z, :, :]
    elif image.ndim == 4:
        xy_image = image[c, z, :, :]
    else:
        xy_image = image[r, c, z, :, :]

    # plot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)
    if not rescale and not contrast:
        vmin, vmax = get_minmax_values(image)
        plt.imshow(xy_image, vmin=vmin, vmax=vmax)
    elif rescale and not contrast:
        plt.imshow(xy_image)
    else:
        if xy_image.dtype not in [np.int64, bool]:
            xy_image = utils.rescale(xy_image, channel_to_stretch=0)
        plt.imshow(xy_image)
    if title is not None and not remove_frame:
        plt.title(title, fontweight="bold", fontsize=25)
    if not remove_frame:
        plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return


def plot_images(images, rescale=False, contrast=False, titles=None,
                framesize=(15, 5), remove_frame=True, path_output=None,
                ext="png", show=True):
    """Plot or subplot of 2-d images.

    Parameters
    ----------
    images : np.ndarray or List[np.ndarray]
        Images with shape (y, x).
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    titles : List[str]
        Titles of the subplots.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    remove_frame : bool
        Remove axes and frame.
    path_output : str
        Path to save the image (without extension).
    ext : str or List[str]
        Extension used to save the plot. If it is a list of strings, the plot
        will be saved several times.
    show : bool
        Show the figure or not.

    Returns
    -------

    """
    # enlist image if necessary
    if isinstance(images, np.ndarray):
        images = [images]

    # check parameters
    utils.check_parameter(images=list,
                          rescale=bool,
                          contrast=bool,
                          titles=(str, list, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)

    for image in images:
        utils.check_array(image,
                          ndim=2,
                          dtype=[np.uint8, np.uint16, np.int64,
                                 np.float32, np.float64,
                                 bool])

    # we plot 3 images by row maximum
    nrow = int(np.ceil(len(images)/3))
    ncol = min(len(images), 3)

    # plot one image
    if len(images) == 1:
        if titles is not None:
            title = titles[0]
        else:
            title = None
        plot_yx(images[0],
                rescale=rescale,
                contrast=contrast,
                title=title,
                framesize=framesize,
                remove_frame=remove_frame,
                path_output=path_output,
                ext=ext,
                show=show)

        return

    # plot multiple images
    fig, ax = plt.subplots(nrow, ncol, figsize=framesize)

    # one row
    if len(images) in [2, 3]:
        for i, image in enumerate(images):
            if remove_frame:
                ax[i].axis("off")
            if not rescale and not contrast:
                vmin, vmax = get_minmax_values(image)
                ax[i].imshow(image, vmin=vmin, vmax=vmax)
            elif rescale and not contrast:
                ax[i].imshow(image)
            else:
                if image.dtype not in [np.int64, bool]:
                    image = utils.rescale(image, channel_to_stretch=0)
                ax[i].imshow(image)
            if titles is not None:
                ax[i].set_title(titles[i], fontweight="bold", fontsize=10)

    # several rows
    else:
        # we complete the row with empty frames
        r = nrow * 3 - len(images)
        images_completed = [image for image in images] + [None] * r

        for i, image in enumerate(images_completed):
            row = i // 3
            col = i % 3
            if image is None:
                ax[row, col].set_visible(False)
                continue
            if remove_frame:
                ax[row, col].axis("off")
            if not rescale and not contrast:
                vmin, vmax = get_minmax_values(image)
                ax[row, col].imshow(image, vmin=vmin, vmax=vmax)
            elif rescale and not contrast:
                ax[row, col].imshow(image)
            else:
                if image.dtype not in [np.int64, bool]:
                    image = utils.rescale(image, channel_to_stretch=0)
                ax[row, col].imshow(image)
            if titles is not None:
                ax[row, col].set_title(titles[i],
                                       fontweight="bold", fontsize=10)

    plt.tight_layout()
    if path_output is not None:
        save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return
