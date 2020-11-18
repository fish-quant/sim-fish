# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
The simfish.plot subpackage includes functions to plot images.
"""

from .plot_images import plot_yx
from .plot_images import plot_images


_images = [
    "plot_yx",
    "plot_images"]


__all__ = _images
