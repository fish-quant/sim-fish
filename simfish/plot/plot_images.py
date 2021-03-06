# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Functions to plot 2-d pixel and coordinates images.
"""

import matplotlib.pyplot as plt
import numpy as np
import bigfish.stack as stack
import bigfish.plot as plot


# ### Subpixel plot ###

def plot_spots(image, ground_truth=None, prediction=None, subpixel=False,
               rescale=False, contrast=False, title=None, framesize=(8, 8),
               remove_frame=True, path_output=None, ext="png", show=True):
    """Plot spot image with a cross in their localization.

    Parameters
    ----------
    image : np.ndarray
        A 2-d or 3-d image with shape (y, x) or (z, y, x) respectively.
    ground_truth : np.ndarray or None
        Ground truth array with shape (nb_spots, 6) or (nb_spots, 4).
        - coordinate_z (optional)
        - coordinate_y
        - coordinate_x
        - sigma_z (optional)
        - sigma_yx
        - amplitude
    prediction : np.ndarray or None
        Predicted localization array with shape (nb_spots, 3) or (nb_spots, 2).
        - coordinate_z (optional)
        - coordinate_y
        - coordinate_x
    subpixel : bool
        Adapt figure frame to subpixel coordinates.
    rescale : bool
        Rescale pixel values of the image (made by default in matplotlib).
    contrast : bool
        Contrast image.
    title : str or None
        Title of the image.
    framesize : tuple
        Size of the frame used to plot with 'plt.figure(figsize=framesize)'.
    remove_frame : bool
        Remove axes and frame.
    path_output : str or None
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
    stack.check_array(image,
                      ndim=[2, 3],
                      dtype=[np.uint8, np.uint16, np.int64,
                             np.float32, np.float64,
                             bool])
    if ground_truth is not None:
        stack.check_array(ground_truth, ndim=2)
    if prediction is not None:
        stack.check_array(prediction, ndim=2)
    stack.check_parameter(subpixel=bool,
                          rescale=bool,
                          contrast=bool,
                          title=(str, type(None)),
                          framesize=tuple,
                          remove_frame=bool,
                          path_output=(str, type(None)),
                          ext=(str, list),
                          show=bool)

    # get dimension and adapt coordinates
    ndim = len(image.shape)
    if ground_truth is not None:
        gt = ground_truth.copy().astype(np.float64)
    else:
        gt = np.array([], dtype=np.float64).reshape((0, ndim))
    if prediction is not None:
        pred = prediction.copy().astype(np.float64)
    else:
        pred = np.array([], dtype=np.float64).reshape((0, ndim))
    if ndim == 3:
        image = image.max(axis=0)
        gt = gt[:, 1:3]
        pred = pred[:, 1:3]

    # prepare ticks
    if subpixel:
        gt -= 0.5
        pred -= 0.5
        extent = None
        y_ticks = None
        x_ticks = None
    else:
        centers = [0, image.shape[0] - 1, image.shape[1] - 1, 0]
        dy, = - np.diff(centers[2:]) / (image.shape[0] - 1)
        dx, = np.diff(centers[:2]) / (image.shape[1] - 1)
        extent = [centers[0] - dx / 2,
                  centers[1] + dx / 2,
                  centers[2] + dy / 2,
                  centers[3] - dy / 2]
        y_ticks = np.arange(centers[3], centers[2] + dy, dy)
        x_ticks = np.arange(centers[0], centers[1] + dx, dx)

    # initialize plot
    if remove_frame:
        fig = plt.figure(figsize=framesize, frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
    else:
        plt.figure(figsize=framesize)

    # plot image
    if not rescale and not contrast:
        vmin, vmax = plot.get_minmax_values(image)
        plt.imshow(image, vmin=vmin, vmax=vmax, extent=extent)
    elif rescale and not contrast:
        plt.imshow(image, extent=extent)
    else:
        if image.dtype not in [np.int64, bool]:
            image = stack.rescale(image, channel_to_stretch=0)
        plt.imshow(image, extent=extent)

    # plot localizations
    plt.scatter(gt[:, 1], gt[:, 0], color="blue", marker="x")
    plt.scatter(pred[:, 1], pred[:, 0], color="red", marker="x")

    # format plot
    if subpixel:
        plt.ylim((image.shape[0] - 0.5, -0.5))
        plt.xlim((-0.5, image.shape[1] - 0.5))
    else:
        plt.yticks(y_ticks)
        plt.xticks(x_ticks)
    if title is not None:
        plt.title(title, fontweight="bold", fontsize=25)
    if not remove_frame:
        plt.tight_layout()
    if path_output is not None:
        plot.save_plot(path_output, ext)
    if show:
        plt.show()
    else:
        plt.close()

    return
