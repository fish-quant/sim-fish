#!/bin/bash

echo 'Running simulate_spots_subpixel_noise.sh...'

# directories
output_directory='/Users/arthur/output/2021_simulations'

# python script
script='/Users/arthur/sim-fish/scripts/python/simulate_range_noise.py'

# ### 10 spots (subpixel) | 0.05 - 0.4 noise ###

# parameters
experiment="subpixel_all"
n_images=100
n_spots=10
random_n_spots=0
n_clusters=0
random_n_clusters=0
n_spots_cluster=0
random_n_spots_cluster=0
centered_cluster=0
image_shape="(16, 32, 32)"
subpixel_factors="(10, 10, 10)"
voxel_size="(100, 100, 100)"
sigma="(100, 100, 100)"
random_sigma=0.
amplitude=1000
noise_level=300
random_min=0.05
random_max=0.4

python "$script" "$output_directory" "$experiment" "$n_images" \
       "$n_spots" "$random_n_spots" \
       "$n_clusters" "$random_n_clusters" \
       "$n_spots_cluster" "$random_n_spots_cluster" "$centered_cluster" \
       "$image_shape" "$subpixel_factors" "$voxel_size" \
       "$sigma" "$random_sigma" \
       "$amplitude" "$noise_level" "$random_min" "$random_max"