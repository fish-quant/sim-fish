#!/bin/bash

#SBATCH --export=ALL
#SBATCH -J simfish
#SBATCH --exclude=node28,node02
#SBATCH -o /cbio/donnees/aimbert/logs/log-%A.log
#SBATCH -e /cbio/donnees/aimbert/logs/log-%A.err
#SBATCH -t 0-100:00             # Time (DD-HH:MM)
#SBATCH --mem 0                 # Memory per node in MB (0 allocates all the memory)
#SBATCH --ntasks=1              # Number of processes to run (default is 1)
#SBATCH --cpus-per-task=16      # CPU cores per process (default 1)
#SBATCH -p cpu                  # Name of the partition to use

echo 'Running simulate_spots_subpixel.sh...'

echo "SLURM_JOBID: " $SLURM_JOBID

# directories
output_directory='/mnt/data3/aimbert/output/2021_simulations'

# python script
script='/cbio/donnees/aimbert/sim-fish/scripts/python/simulate_range_noise.py'

# ### Subpixel 0.05 - 0.4 ###

# parameters
experiment="subpixel_all"
n_images=100
image_z=32
image_y=32
image_x=32
subpixel_factors_z=20
subpixel_factors_y=20
subpixel_factors_x=20
voxel_size_z=100
voxel_size_yx=100
n_spots=10
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
script='/cbio/donnees/aimbert/sim-fish/scripts/python/simulate_range_spots.py'

# ### Subpixel 0.065 ###

# parameters
experiment="subpixel_065"
n_images=100
image_z=32
image_y=32
image_x=32
subpixel_factors_z=20
subpixel_factors_y=20
subpixel_factors_x=20
voxel_size_z=100
voxel_size_yx=100
n_spots_min=1
n_spots_max=15
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

# ### Subpixel 0.135 ###

# parameters
experiment="subpixel_135"
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

# ### Subpixel 0.400 ###

# parameters
experiment="subpixel_400"
random_amplitude=0.400
random_noise=0.400

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