#!/bin/bash

echo 'Running simulate_spots.sh...'

# directories
output_directory='/Users/arthur/output/2021_simulations'

# python script
script='/Users/arthur/sim-fish/scripts/python/simulate_range_noise.py'

# ### Spots 0.05 - 0.4 ###

# parameters
experiment="spots_all"
n_images=100
image_z=32
image_y=128
image_x=128
subpixel_factors_z=10
subpixel_factors_y=10
subpixel_factors_x=10
voxel_size_z=100
voxel_size_yx=100
n_spots=100
random_n_spots=0
n_clusters=0
random_n_clusters=0
n_spots_cluster=0
random_n_spots_cluster=0
centered_cluster=0
sigma_z=100
sigma_yx=100
random_sigma=0.
amplitude=1000
noise_level=300
random_min=0.05
random_max=0.4

python "$script" "$output_directory" "$experiment" "$n_images" \
        "$image_z" "$image_y" "$image_x" \
        "$subpixel_factors_z" "$subpixel_factors_y" "$subpixel_factors_x" \
        "$voxel_size_z" "$voxel_size_yx" \
        "$n_spots" "$random_n_spots" \
        "$n_clusters" "$random_n_clusters" \
        "$n_spots_cluster" "$random_n_spots_cluster" "$centered_cluster" \
        "$sigma_z" "$sigma_yx" "$random_sigma" \
        "$amplitude" \
        "$noise_level" "$random_min" "$random_max"

# python script
script='/Users/arthur/sim-fish/scripts/python/simulate_range_spots.py'

# ### Spots 0.065 ###

# parameters
experiment="spots_065"
n_images=100
image_z=32
image_y=128
image_x=128
subpixel_factors_z=10
subpixel_factors_y=10
subpixel_factors_x=10
voxel_size_z=100
voxel_size_yx=100
n_spots_min=1
n_spots_max=300
random_n_spots=0
n_clusters=0
random_n_clusters=0
n_spots_cluster=0
random_n_spots_cluster=0
sigma_z=100
sigma_yx=100
random_sigma=0.
amplitude=1000
random_amplitude=0.065
noise_level=300
random_noise=0.065

python "$script" "$output_directory" "$experiment" "$n_images" \
        "$image_z" "$image_y" "$image_x" \
        "$subpixel_factors_z" "$subpixel_factors_y" "$subpixel_factors_x" \
        "$voxel_size_z" "$voxel_size_yx" \
        "$n_spots_min" "$n_spots_max" "$random_n_spots" \
        "$n_clusters" "$random_n_clusters" \
        "$n_spots_cluster" "$random_n_spots_cluster" \
        "$sigma_z" "$sigma_yx" "$random_sigma" \
        "$amplitude" "$random_amplitude" \
        "$noise_level" "$random_noise"

# ### Spots 0.135 ###

# parameters
experiment="spots_135"
random_amplitude=0.135
random_noise=0.135

python "$script" "$output_directory" "$experiment" "$n_images" \
        "$image_z" "$image_y" "$image_x" \
        "$subpixel_factors_z" "$subpixel_factors_y" "$subpixel_factors_x" \
        "$voxel_size_z" "$voxel_size_yx" \
        "$n_spots_min" "$n_spots_max" "$random_n_spots" \
        "$n_clusters" "$random_n_clusters" \
        "$n_spots_cluster" "$random_n_spots_cluster" \
        "$sigma_z" "$sigma_yx" "$random_sigma" \
        "$amplitude" "$random_amplitude" \
        "$noise_level" "$random_noise"

# ### Spots 0.400 ###

# parameters
experiment="spots_400"
random_amplitude=0.4
random_noise=0.4

python "$script" "$output_directory" "$experiment" "$n_images" \
        "$image_z" "$image_y" "$image_x" \
        "$subpixel_factors_z" "$subpixel_factors_y" "$subpixel_factors_x" \
        "$voxel_size_z" "$voxel_size_yx" \
        "$n_spots_min" "$n_spots_max" "$random_n_spots" \
        "$n_clusters" "$random_n_clusters" \
        "$n_spots_cluster" "$random_n_spots_cluster" \
        "$sigma_z" "$sigma_yx" "$random_sigma" \
        "$amplitude" "$random_amplitude" \
        "$noise_level" "$random_noise"