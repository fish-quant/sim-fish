#!/bin/bash

echo 'Running simulate_spots.sh...'

# directories
output_directory='/Users/arthur/output/2021_simulations'

# python script
script='/Users/arthur/sim-fish/scripts/simulate_spots.py'

# parameters
experiment="test"
n_images=100
image_z=10
image_y=256
image_x=256
subpixel_factors_z=10
subpixel_factors_y=10
subpixel_factors_x=10
voxel_size_z=100
voxel_size_yx=100
n_spots_min=50
n_spots_max=300
random_n_spots=1
n_clusters=3
random_n_clusters=1
n_spots_cluster=10
sigma_z=100
sigma_yx=100
random_sigma=0.05
amplitude=5000
random_amplitude=0.05
noise_level=300
random_noise=0.05

python "$script" "$output_directory" "$experiment" "$n_images" \
        "$image_z" "$image_y" "$image_x" \
        "$subpixel_factors_z" "$subpixel_factors_y" "$subpixel_factors_x" \
        "$voxel_size_z" "$voxel_size_yx" \
        "$n_spots_min" "$n_spots_max" "$random_n_spots" \
        "$n_clusters" "$random_n_clusters" "$n_spots_cluster" \
        "$sigma_z" "$sigma_yx" "$random_sigma" \
        "$amplitude" "$random_amplitude" \
        "$noise_level" "$random_noise"