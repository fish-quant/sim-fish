.. _utils overview:

Utility functions
*****************

.. currentmodule:: simfish

To simulate patterns consistent with cell morphology  we use actual cell
templates. The dataset is available `here <https://doi.org/10.5281/zenodo.
6106718>`_. This dataset includes 318 shapes (cell, nucleus and protrusion
coordinates) we can used to build an authentic cell morphology. Precomputed
distance maps and an index file saved as CSV are also available.

Download and extract the template dataset:

* :func:`simfish.load_extract_template`

Load an index dataframe that summarize templates metadata:

* :func:`simfish.read_index_template`

Load and build templates:

* :func:`simfish.build_templates`
* :func:`simfish.build_template`

.. autofunction:: load_extract_template
.. autofunction:: read_index_template
.. autofunction:: build_templates
.. autofunction:: build_template
