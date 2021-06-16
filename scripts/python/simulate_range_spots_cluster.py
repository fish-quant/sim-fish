# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Script to simulate images with different number of spots per cluster.
"""

import os
import argparse
import sys

import numpy as np
import bigfish.stack as stack
import simfish.spots as spots
import simfish.plot as plot

from joblib import Parallel, delayed


class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


if __name__ == "__main__":
    print()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("output_directory",
                        help="Path of the output directory.",
                        type=str)
    parser.add_argument("experiment",
                        help="Name of the experiment.",
                        type=str)
    parser.add_argument("n_images",
                        help="Number of images to simulate.",
                        type=int,
                        default=100)
    parser.add_argument("image_z",
                        help="Image shape (z axis).",
                        type=int,
                        default=10)
    parser.add_argument("image_y",
                        help="Image shape (y axis).",
                        type=int,
                        default=256)
    parser.add_argument("image_x",
                        help="Image shape (x axis).",
                        type=int,
                        default=256)
    parser.add_argument("subpixel_factors_z",
                        help="Multiplicative factor to simulate subpixel "
                             "accuracy along z axis.",
                        type=int,
                        default=10)
    parser.add_argument("subpixel_factors_y",
                        help="Multiplicative factor to simulate subpixel "
                             "accuracy along y axis.",
                        type=int,
                        default=10)
    parser.add_argument("subpixel_factors_x",
                        help="Multiplicative factor to simulate subpixel "
                             "accuracy along x axis.",
                        type=int,
                        default=10)
    parser.add_argument("voxel_size_z",
                        help="Voxel size along the z axis (in nanometer).",
                        type=int,
                        default=100)
    parser.add_argument("voxel_size_yx",
                        help="Voxel size along the yx axis (in nanometer).",
                        type=int,
                        default=100)
    parser.add_argument("n_spots",
                        help="Number of spots to simulate per image.",
                        type=int,
                        default=100)
    parser.add_argument("random_n_spots",
                        help="Randomly sample number of spots to simulate.",
                        type=int,
                        default=1)
    parser.add_argument("n_clusters",
                        help="Number of clusters to simulate per image.",
                        type=int,
                        default=3)
    parser.add_argument("random_n_clusters",
                        help="Randomly sample number of clusters to simulate.",
                        type=int,
                        default=1)
    parser.add_argument("n_spots_cluster_min",
                        help="Number of spots to simulate per cluster "
                             "(lower bound).",
                        type=int,
                        default=3)
    parser.add_argument("n_spots_cluster_max",
                        help="Number of spots to simulate per cluster "
                             "(upper bound).",
                        type=int,
                        default=20)
    parser.add_argument("random_n_spots_cluster",
                        help="Random number of spots to simulate per cluster.",
                        type=int,
                        default=1)
    parser.add_argument("centered_cluster",
                        help="Center cluster (if only one cluster simulated).",
                        type=int,
                        default=1)
    parser.add_argument("sigma_z",
                        help="PSF standard deviation along the z axis "
                             "(in nanometer).",
                        type=int,
                        default=100)
    parser.add_argument("sigma_yx",
                        help="PSF standard deviation along the yx axis "
                             "(in nanometer).",
                        type=int,
                        default=100)
    parser.add_argument("random_sigma",
                        help="Random margin over the sigma parameters.",
                        type=float,
                        default=0.0)
    parser.add_argument("amplitude",
                        help="Average maximum pixel intensity of the spots.",
                        type=int,
                        default=1000)
    parser.add_argument("random_amplitude",
                        help="Random margin over the amplitude parameter..",
                        type=float,
                        default=0.05)
    parser.add_argument("noise_level",
                        help="Noise level in the image.",
                        type=int,
                        default=300)
    parser.add_argument("random_noise",
                        help="Random margin over the noise parameter.",
                        type=float,
                        default=0.05)

    # initialize parameters
    args = parser.parse_args()
    output_directory = args.output_directory
    experiment = args.experiment
    n_images = args.n_images
    image_z = args.image_z
    image_y = args.image_y
    image_x = args.image_x
    if image_z > 0:
        image_shape = (image_z, image_y, image_x)
    else:
        image_shape = (image_y, image_x)
    subpixel_factors_z = args.subpixel_factors_z
    subpixel_factors_y = args.subpixel_factors_y
    subpixel_factors_x = args.subpixel_factors_x
    if subpixel_factors_z > 0:
        subpixel_factors = (subpixel_factors_z, subpixel_factors_y,
                            subpixel_factors_x)
    else:
        subpixel_factors = (subpixel_factors_y, subpixel_factors_x)
    image_dtype = np.uint16
    voxel_size_z = args.voxel_size_z
    voxel_size_yx = args.voxel_size_yx
    n_spots = args.n_spots
    random_n_spots = bool(args.random_n_spots)
    n_clusters = args.n_clusters
    random_n_clusters = bool(args.random_n_clusters)
    n_spots_cluster_min = args.n_spots_cluster_min
    n_spots_cluster_max = args.n_spots_cluster_max
    n_spots_cluster = (n_spots_cluster_min, n_spots_cluster_max)
    random_n_spots_cluster = bool(args.random_n_spots_cluster)
    centered_cluster = bool(args.centered_cluster)
    sigma_z = args.sigma_z
    sigma_yx = args.sigma_yx
    random_sigma = args.random_sigma
    amplitude = args.amplitude
    random_amplitude = args.random_amplitude
    noise_level = args.noise_level
    random_noise = args.random_noise

    # folders
    path_directory = os.path.join(output_directory, experiment)
    if not os.path.exists(path_directory):
        os.mkdir(path_directory)
    path_directory_image = os.path.join(path_directory, "images")
    if not os.path.exists(path_directory_image):
        os.mkdir(path_directory_image)
    path_directory_gt = os.path.join(path_directory, "gt")
    if not os.path.exists(path_directory_gt):
        os.mkdir(path_directory_gt)
    path_directory_plot = os.path.join(path_directory, "plots")
    if not os.path.exists(path_directory_plot):
        os.mkdir(path_directory_plot)

    # save log
    path_log_file = os.path.join(path_directory, "log.txt")
    sys.stdout = Logger(path_log_file)

    # display information
    print("Output directory: {0}".format(output_directory))
    print("Experiment: {0}".format(experiment))
    print("Number of images: {0}".format(n_images))
    print("Image shape: {0}".format(image_shape))
    print("Subpixel factors: {0}".format(subpixel_factors))
    print("Image dtype: {0}".format(image_dtype))
    print("Size voxel z: {0}".format(voxel_size_z))
    print("Size voxel yx: {0}".format(voxel_size_yx))
    print("Number of spots: {0}".format(n_spots))
    print("Random number of spots: {0}".format(random_n_spots))
    print("Number of clusters: {0}".format(n_clusters))
    print("Random number of clusters: {0}".format(random_n_clusters))
    print("Number of spots per cluster (min, max): {0}"
          .format(n_spots_cluster))
    print("Random number of spots per cluster: {0}"
          .format(random_n_spots_cluster))
    print("Center cluster (if only one cluster simulated): {0}"
          .format(centered_cluster))
    print("Sigma z: {0}".format(sigma_z))
    print("Sigma yx: {0}".format(sigma_yx))
    print("Random sigma: {0}".format(random_sigma))
    print("Amplitude: {0}".format(amplitude))
    print("Random amplitude: {0}".format(random_amplitude))
    print("Noise level: {0}".format(noise_level))
    print("Random noise: {0}".format(random_noise))
    print()

    # define number of spots per cluster
    l_n = np.linspace(n_spots_cluster[0], n_spots_cluster[1],
                      num=n_images, dtype=np.int64)

    def fct_to_process(i, n):
        # simulate images
        image, ground_truth = spots.simulate_image(
            image_shape=image_shape,
            image_dtype=image_dtype,
            subpixel_factors=subpixel_factors,
            voxel_size_z=voxel_size_z,
            voxel_size_yx=voxel_size_yx,
            n_spots=n_spots,
            random_n_spots=random_n_spots,
            n_clusters=n_clusters,
            random_n_clusters=random_n_clusters,
            n_spots_cluster=n,
            random_n_spots_cluster=random_n_spots_cluster,
            centered_cluster=centered_cluster,
            sigma_z=sigma_z,
            sigma_yx=sigma_yx,
            random_sigma=random_sigma,
            amplitude=amplitude,
            random_amplitude=random_amplitude,
            noise_level=noise_level,
            random_noise=random_noise)

        # save image
        path = os.path.join(path_directory_image, "image_{0}.tif".format(i))
        stack.save_image(image, path)

        # complete ground truth and save it
        new_column = np.array([n] * len(ground_truth))
        new_column = new_column[:, np.newaxis]
        ground_truth = np.hstack([ground_truth, new_column])
        path = os.path.join(path_directory_gt, "gt_{0}.csv".format(i))
        stack.save_data_to_csv(ground_truth, path)

        # plot
        path = os.path.join(path_directory_plot, "plot_{0}.png".format(i))
        subpixel = True if subpixel_factors is not None else False
        plot.plot_spots(
            image,
            ground_truth=None,
            prediction=None,
            subpixel=subpixel,
            rescale=True,
            contrast=False,
            title="Number of mRNAs: {0}".format(len(ground_truth)),
            framesize=(8, 8),
            remove_frame=False,
            path_output=path,
            ext="png",
            show=False)

        return

    # parallelization
    Parallel(n_jobs=4)(delayed(fct_to_process)(i, int(n))
                       for i, n in enumerate(l_n))

    print()
    print("Script done!")
