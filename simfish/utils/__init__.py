# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The simfish.utils subpackage includes functions to read, save and preprocess
images.
"""

from .utils import check_array
from .utils import check_df
from .utils import check_parameter
from .utils import check_range_value
from .utils import rescale
from .utils import cast_img_uint8
from .utils import cast_img_uint16
from .utils import cast_img_float32
from .utils import cast_img_float64

from .io import read_image
from .io import read_array
from .io import read_array_from_csv
from .io import read_dataframe_from_csv
from .io import save_image
from .io import save_array
from .io import save_data_to_csv


_utils = [
    "check_array",
    "check_df",
    "check_parameter",
    "check_range_value",
    "rescale",
    "cast_img_uint8",
    "cast_img_uint16",
    "cast_img_float32",
    "cast_img_float64"]

_io = [
    "read_image",
    "read_array",
    "read_array_from_csv",
    "read_dataframe_from_csv",
    "save_image",
    "save_array",
    "save_data_to_csv"]


__all__ = _utils + _io
