# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The simfish package includes function to simulate RNA spots and FISH images in
2-d and 3-d.
"""

# keep a MAJOR.MINOR.PATCH format
# MAJOR: major API changes
# MINOR: new features
# PATCH: backwards compatible bug fixes
# MAJOR.MINOR.PATCHdev means a version under development
__version__ = "0.1.0dev"


from .utils import read_index_template
from .utils import build_templates
from .utils import build_template

from .pattern_simulation import simulate_ground_truth
from .pattern_simulation import get_random_probability_map
from .pattern_simulation import get_random_out_probability_map
from .pattern_simulation import get_random_in_probability_map
from .pattern_simulation import get_nuclear_edge_probability_map
from .pattern_simulation import get_perinuclear_probability_map
from .pattern_simulation import get_cell_edge_probability_map
from .pattern_simulation import get_protrusion_probability_map
from .pattern_simulation import build_probability_map
from .pattern_simulation import simulate_localization_pattern


from .spot_simulation import add_spots

from .noise_simulation import add_white_noise

from .image_simulation import simulate_images
from .image_simulation import simulate_image


_utils = [
    "read_index_template",
    "build_templates",
    "build_template"]

_patterns = [
    "simulate_ground_truth",
    "get_random_probability_map",
    "get_random_out_probability_map",
    "get_random_in_probability_map",
    "get_nuclear_edge_probability_map",
    "get_perinuclear_probability_map",
    "get_cell_edge_probability_map",
    "get_protrusion_probability_map",
    "build_probability_map",
    "simulate_localization_pattern"]

_spots = [
    "add_spots"]

_noise = [
    "add_white_noise"]

_images = [
    "simulate_images",
    "simulate_image"]

__all__ = _patterns + _spots + _noise + _images

# TODO complete documentation
# TODO add colocalization
# TODO add segmentation mask
# TODO build package and push to pypi
