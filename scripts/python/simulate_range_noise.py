# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Script to simulate images with different noise variance.
"""

import os
import argparse
import sys

import numpy as np
import bigfish.stack as stack
import bigfish.plot as plot
import simfish as sim

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
    parser.add_argument("n_spots_cluster",
                        help="Number of spots to simulate per cluster.",
                        type=int,
                        default=10)
    parser.add_argument("random_n_spots_cluster",
                        help="Random number of spots to simulate per cluster.",
                        type=int,
                        default=1)
    parser.add_argument("centered_cluster",
                        help="Center cluster (if only one cluster simulated).",
                        type=int,
                        default=1)
    parser.add_argument("image_shape",
                        help="Image shape.",
                        type=tuple,
                        default=(10, 256, 256))
    parser.add_argument("subpixel_factors",
                        help="Multiplicative factor to simulate subpixel "
                             "accuracy along along each dimension.",
                        type=tuple,
                        default=(10, 10, 10))
    parser.add_argument("voxel_size",
                        help="Voxel size (in nanometer).",
                        type=tuple,
                        default=(100, 100, 100))
    parser.add_argument("sigma",
                        help="PSF standard deviation (in nanometer).",
                        type=tuple,
                        default=(100, 100, 100))
    parser.add_argument("random_sigma",
                        help="Random margin over the sigma parameters.",
                        type=float,
                        default=0.0)
    parser.add_argument("amplitude",
                        help="Average maximum pixel intensity of the spots.",
                        type=int,
                        default=1000)
    parser.add_argument("noise_level",
                        help="Noise level in the image.",
                        type=int,
                        default=300)
    parser.add_argument("random_min",
                        help="Random margin over the noise and amplitude "
                             "parameters (lower bound).",
                        type=float,
                        default=0.05)
    parser.add_argument("random_max",
                        help="Random margin over the noise and amplitude "
                             "parameters (upper bound).",
                        type=float,
                        default=0.40)

    # initialize parameters
    args = parser.parse_args()
    output_directory = args.output_directory
    experiment = args.experiment
    n_images = args.n_images
    n_spots = args.n_spots
    random_n_spots = bool(args.random_n_spots)
    n_clusters = args.n_clusters
    random_n_clusters = bool(args.random_n_clusters)
    n_spots_cluster = args.n_spots_cluster
    random_n_spots_cluster = bool(args.random_n_spots_cluster)
    centered_cluster = bool(args.centered_cluster)
    image_dtype = np.uint16
    image_shape = args.image_shape
    ndim = len(image_shape)
    subpixel_factors = args.subpixel_factors
    voxel_size = args.voxel_size
    sigma = args.sigma
    random_sigma = args.random_sigma
    amplitude = args.amplitude
    noise_level = args.noise_level
    random_margin_min = args.random_min
    random_margin_max = args.random_max
    random_margin = (random_margin_min, random_margin_max)

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
    print("Number of spots: {0}".format(n_spots))
    print("Random number of spots: {0}".format(random_n_spots))
    print("Number of clusters: {0}".format(n_clusters))
    print("Random number of clusters: {0}".format(random_n_clusters))
    print("Number of spots per cluster: {0}".format(n_spots_cluster))
    print("Random number of spots per cluster: {0}"
          .format(random_n_spots_cluster))
    print("Center cluster (if only one cluster simulated): {0}"
          .format(centered_cluster))
    print("Image dtype: {0}".format(image_dtype))
    print("Image shape: {0}".format(image_shape))
    print("Subpixel factors: {0}".format(subpixel_factors))
    print("Size voxel: {0}".format(voxel_size))
    print("Sigma: {0}".format(sigma))
    print("Random sigma: {0}".format(random_sigma))
    print("Amplitude: {0}".format(amplitude))
    print("Noise level: {0}".format(noise_level))
    print("Random margin (min, max): {0}".format(random_margin))
    print()

    # define noise variance (log scale)
    random_margin_min_ = np.log(random_margin[0]) / np.log(10)
    random_margin_max_ = np.log(random_margin[1]) / np.log(10)
    l_n = np.logspace(random_margin_min_, random_margin_max_, num=n_images)

    def fct_to_process(i, n):
        # simulate images
        image, ground_truth = sim.simulate_image(
            ndim=ndim,
            n_spots=n_spots,
            random_n_spots=random_n_spots,
            n_clusters=n_clusters,
            random_n_clusters=random_n_clusters,
            n_spots_cluster=n_spots_cluster,
            random_n_spots_cluster=random_n_spots_cluster,
            centered_cluster=centered_cluster,
            image_shape=image_shape,
            image_dtype=np.uint16,
            subpixel_factors=subpixel_factors,
            voxel_size=voxel_size,
            sigma=sigma,
            random_sigma=random_sigma,
            amplitude=amplitude,
            random_amplitude=n,
            noise_level=noise_level,
            random_noise=n)

        # save image and ground truth
        path = os.path.join(path_directory_image, "image_{0}.tif".format(i))
        stack.save_image(image, path)
        path = os.path.join(path_directory_gt, "gt_{0}.csv".format(i))
        stack.save_data_to_csv(ground_truth, path)

        # plot
        path = os.path.join(path_directory_plot, "plot_{0}.png".format(i))
        plot.plot_images(
            images=image,
            rescale=True,
            titles="Number of spots: {0}".format(len(ground_truth)),
            framesize=(8, 8),
            remove_frame=False,
            path_output=path,
            show=False)

        return

    # parallelization
    Parallel(n_jobs=4)(delayed(fct_to_process)(i, n)
                       for i, n in enumerate(l_n))

    print()
    print("Script done!")
