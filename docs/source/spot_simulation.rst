.. _spot_simulation overview:

Spot simulation
***************

.. currentmodule:: simfish.spots

Functions used to simulate spot coordinates and image, in 2D and 3D.

Simulation is performed in three steps:

* We simulate ground truth coordinates.

    * :func:`simfish.spots.simulate_ground_truth`

* From an empty frame we simulate a gaussian signal in every spot location.

    * :func:`simfish.spots.add_spots`

* We add an additive background noise in image if needed.

    * :func:`simfish.spots.add_white_noise`

The full process can be run with:

* :func:`simfish.spots.simulate_images`
* :func:`simfish.spots.simulate_image`

.. autofunction:: simulate_ground_truth
.. autofunction:: add_spots
.. autofunction:: add_white_noise
.. autofunction:: simulate_images
.. autofunction:: simulate_image