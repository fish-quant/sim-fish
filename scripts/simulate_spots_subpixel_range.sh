#!/bin/bash

# Simulate images with a range of spots and a subpixel accuracy

echo 'Running simulate_spots_subpixel_range.sh...'

# directories
output_directory='/Users/arthur/output/2021_simulations'

# python script
script='/Users/arthur/sim-fish/scripts/python/simulate_range_spots.py'

# ### 1 - 15 spots (subpixel) | 0.065 noise ###

# parameters
experiment="subpixel_065"
n_images=100
n_spots_min=1
n_spots_max=15
random_n_spots=0
n_clusters=0
random_n_clusters=0
n_spots_cluster=0
random_n_spots_cluster=0
image_shape="(16, 32, 32)"
subpixel_factors="(10, 10, 10)"
voxel_size="(100, 100, 100)"
sigma="(100, 100, 100)"
random_sigma=0.
amplitude=1000
random_amplitude=0.065
noise_level=300
random_noise=0.065

python "$script" "$output_directory" "$experiment" "$n_images" \
       "$n_spots_min" "$n_spots_max" "$random_n_spots" \
       "$n_clusters" "$random_n_clusters" \
       "$n_spots_cluster" "$random_n_spots_cluster" \
       "$image_shape" "$subpixel_factors" "$voxel_size" \
       "$sigma"  "$random_sigma" \
       "$amplitude" "$random_amplitude" \
       "$noise_level" "$random_noise"

# ### 1 - 15 spots (subpixel) | 0.135 noise ###

# parameters
experiment="subpixel_135"
random_amplitude=0.135
random_noise=0.135

python "$script" "$output_directory" "$experiment" "$n_images" \
       "$n_spots_min" "$n_spots_max" "$random_n_spots" \
       "$n_clusters" "$random_n_clusters" \
       "$n_spots_cluster" "$random_n_spots_cluster" \
       "$image_shape" "$subpixel_factors" "$voxel_size" \
       "$sigma"  "$random_sigma" \
       "$amplitude" "$random_amplitude" \
       "$noise_level" "$random_noise"

# ### 1 - 15 spots (subpixel) | 0.400 noise ###

# parameters
experiment="subpixel_400"
random_amplitude=0.400
random_noise=0.400

python "$script" "$output_directory" "$experiment" "$n_images" \
       "$n_spots_min" "$n_spots_max" "$random_n_spots" \
       "$n_clusters" "$random_n_clusters" \
       "$n_spots_cluster" "$random_n_spots_cluster" \
       "$image_shape" "$subpixel_factors" "$voxel_size" \
       "$sigma"  "$random_sigma" \
       "$amplitude" "$random_amplitude" \
       "$noise_level" "$random_noise"