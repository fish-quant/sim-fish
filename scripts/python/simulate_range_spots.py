# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Script to simulate images with different number of spots.
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


def cast_tuple_int(tuple_str):
    tuple_str = tuple_str[1:-1].split(",")
    tuple_int = tuple([int(x) for x in tuple_str])

    return tuple_int


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
    parser.add_argument("n_spots_min",
                        help="Number of spots to simulate per image "
                             "(lower bound).",
                        type=int,
                        default=50)
    parser.add_argument("n_spots_max",
                        help="Number of spots to simulate per image "
                             "(upper bound).",
                        type=int,
                        default=300)
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
    parser.add_argument("image_shape",
                        help="Image shape.",
                        type=str,
                        default="(10, 256, 256)")
    parser.add_argument("subpixel_factors",
                        help="Multiplicative factor to simulate subpixel "
                             "accuracy along along each dimension.",
                        type=str,
                        default="(1, 1, 1)")
    parser.add_argument("voxel_size",
                        help="Voxel size (in nanometer).",
                        type=str,
                        default="(100, 100, 100)")
    parser.add_argument("sigma",
                        help="PSF standard deviation (in nanometer).",
                        type=str,
                        default="(100, 100, 100)")
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
    n_spots_min = args.n_spots_min
    n_spots_max = args.n_spots_max
    n_spots = (n_spots_min, n_spots_max)
    random_n_spots = bool(args.random_n_spots)
    n_clusters = args.n_clusters
    random_n_clusters = bool(args.random_n_clusters)
    n_spots_cluster = args.n_spots_cluster
    random_n_spots_cluster = bool(args.random_n_spots_cluster)
    centered_cluster = False
    image_dtype = np.uint16
    image_shape = cast_tuple_int(args.image_shape)
    ndim = len(image_shape)
    subpixel_factors = cast_tuple_int(args.subpixel_factors)
    voxel_size = cast_tuple_int(args.voxel_size)
    sigma = cast_tuple_int(args.sigma)
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
    print("Number of dimensions: {0}".format(ndim))
    print("Number of spots (min, max): {0}".format(n_spots))
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
    print("Random amplitude: {0}".format(random_amplitude))
    print("Noise level: {0}".format(noise_level))
    print("Random noise: {0}".format(random_noise))
    print()

    # define number of spots
    l_n = np.linspace(n_spots[0], n_spots[1], num=n_images, dtype=np.int64)

    def fct_to_process(i, n):
        # simulate images
        image, ground_truth = sim.simulate_image(
            ndim=ndim,
            n_spots=n,
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
            random_amplitude=random_amplitude,
            noise_level=noise_level,
            random_noise=random_noise)

        # save image and ground truth
        path = os.path.join(path_directory_image, "image_{0}.tif".format(i))
        stack.save_image(image, path)
        path = os.path.join(path_directory_gt, "gt_{0}.csv".format(i))
        stack.save_data_to_csv(ground_truth, path)

        # plot
        path = os.path.join(path_directory_plot, "plot_{0}.png".format(i))
        image_mip = stack.maximum_projection(image)
        plot.plot_images(
            images=image_mip,
            rescale=True,
            titles=["Number of spots: {0}".format(len(ground_truth))],
            framesize=(8, 8),
            remove_frame=False,
            path_output=path,
            show=False)

        return

    # parallelization
    Parallel(n_jobs=4)(delayed(fct_to_process)(i, int(n))
                       for i, n in enumerate(l_n))

    print()
    print("Script done!")
