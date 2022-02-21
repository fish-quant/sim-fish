#!/bin/bash

# Simulate images of clustered spots with a range of cluster size

echo 'Running simulate_cluster_range.sh...'

# directories
output_directory='/Users/arthur/output/2021_simulations'

# python script
script='/Users/arthur/sim-fish/scripts/python/simulate_range_spots_cluster.py'

# ### 3 - 20 spots per cluster | 0.065 noise ###

# parameters
experiment="cluster_065"
n_images=100
n_spots=100
random_n_spots=0
n_clusters=1
random_n_clusters=0
n_spots_cluster_min=3
n_spots_cluster_max=20
random_n_spots_cluster=0
centered_cluster=1
image_shape="(32, 128, 128)"
subpixel_factors="(1, 1, 1)"
voxel_size="(100, 100, 100)"
sigma="(100, 100, 100)"
random_sigma=0.
amplitude=1000
random_amplitude=0.065
noise_level=300
random_noise=0.065

python "$script" "$output_directory" "$experiment" "$n_images" \
       "$n_spots" "$random_n_spots" \
       "$n_clusters" "$random_n_clusters" \
       "$n_spots_cluster_min" "$n_spots_cluster_max" \
       "$random_n_spots_cluster" \ "$centered_cluster" \
       "$image_shape" "$subpixel_factors" "$voxel_size" \
       "$sigma" "$random_sigma" \
       "$amplitude" "$random_amplitude" \
       "$noise_level" "$random_noise"

# ### 3 - 20 spots per cluster | 0.135 noise ###

# parameters
experiment="cluster_135"
random_amplitude=0.135
random_noise=0.135

python "$script" "$output_directory" "$experiment" "$n_images" \
       "$n_spots" "$random_n_spots" \
       "$n_clusters" "$random_n_clusters" \
       "$n_spots_cluster_min" "$n_spots_cluster_max" \
       "$random_n_spots_cluster" \ "$centered_cluster" \
       "$image_shape" "$subpixel_factors" "$voxel_size" \
       "$sigma" "$random_sigma" \
       "$amplitude" "$random_amplitude" \
       "$noise_level" "$random_noise"

# ### 3 - 20 spots per cluster | 0.400 noise ###

# parameters
experiment="cluster_400"
random_amplitude=0.4
random_noise=0.4

python "$script" "$output_directory" "$experiment" "$n_images" \
       "$n_spots" "$random_n_spots" \
       "$n_clusters" "$random_n_clusters" \
       "$n_spots_cluster_min" "$n_spots_cluster_max" \
       "$random_n_spots_cluster" \ "$centered_cluster" \
       "$image_shape" "$subpixel_factors" "$voxel_size" \
       "$sigma" "$random_sigma" \
       "$amplitude" "$random_amplitude" \
       "$noise_level" "$random_noise"