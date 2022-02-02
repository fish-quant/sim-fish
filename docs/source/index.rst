.. sim-fish documentation main file, created by
   sphinx-quickstart on Thu Nov 19 22:45:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Sim-FISH
********

Getting started
===============

To avoid dependency conflicts, we recommend the the use of a dedicated
`virtual <https://docs.python.org/3.6/library/venv.html>`_ or `conda
<https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-
environments.html>`_ environment.  In a terminal run the command:

.. code-block:: bash

   $ conda create -n simfish_env python=3.6
   $ source activate simfish_env

We recommend two options to then install Sim-FISH in your virtual environment.

Download the package from PyPi
------------------------------

Use the package manager `pip <https://pip.pypa.io/en/stable>`_ to install
Sim-FISH. In a terminal run the command:

.. code-block:: bash

   $ pip install sim-fish

Clone package from Github
-------------------------

Clone the project's `Github repository <https://github.com/fish-quant/sim-
fish>`_ and install it manually with the following commands:

.. code-block:: bash

   $ git clone git@github.com:fish-quant/sim-fish.git
   $ cd sim-fish
   $ pip install .

------------

API reference
*************

.. toctree::
   :caption: Simulations

   spot_simulation
   pattern_simulation

------------

Support
=======

If you have any question relative to the package, please open an `issue
<https://github.com/fish-quant/sim-fish/issues>`_ on Github.