# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for simfish package.
"""

import os
import sys
import zipfile

import numpy as np
import pandas as pd

import bigfish.stack as stack

from urllib.request import urlretrieve


# ### Templates ###

def load_extract_template(path_output, verbose=True):
    """Download template dataset zipfile and extract it.

    Parameters
    ----------
    path_output : str
        Path location to save dataset.
    verbose : bool, default=True
        Show download progression.

    Returns
    -------
    path_final : str
        Path of the downloaded dataset.

    """
    # check parameters
    stack.check_parameter(
        path_output=str,
        verbose=bool)

    # get remote url
    remote_url = "https://zenodo.org/record/6106718/files/templates.zip"

    # get output paths
    path_download = os.path.join(path_output, "templates.zip")
    path_final = os.path.join(path_output, "templates")

    # download and save data
    if verbose:
        urlretrieve(remote_url, path_download, _reporthook)
        print()
    else:
        urlretrieve(remote_url, path_download)

    # extract zipfile
    with zipfile.ZipFile(path_download, 'r') as zip_ref:
        zip_ref.extractall(path_output)

    # remove zipfile
    os.remove(path_download)
    if verbose:
        print("Templates downloaded and ready!")

    return path_final


def _reporthook(count, block_size, total_size):
    if count == 0:
        pass
    else:
        progress_size = int(count * block_size / (1024 * 1024))
        percent = min(int(count * block_size * 100 / total_size), 100)
        sys.stdout.write("\r...{0}% ({1}MB)".format(percent, progress_size))
        sys.stdout.flush()

    return


def read_index_template(path_template_directory):
    """Load and read dataframe with templates metadata.

    Parameters
    ----------
    path_template_directory : str
        Path of the templates directory.

    Returns
    -------
    df : pd.DataFrame
        Dataframe with the templates metadata. Columns are:

        * 'id' instance id.
        * 'shape' shape of the cell image (with the format '{z}_{y}_{x}').
        * 'protrusion_flag' presence or not of protrusion in  the instance.

    """
    # check parameter
    stack.check_parameter(path_template_directory=str)

    # read dataframe
    path = os.path.join(path_template_directory, "index.csv")
    df = stack.read_dataframe_from_csv(path)

    return df


def build_templates(path_template_directory, protrusion=None):
    """Return a generator to simulate several templates.

    Parameters
    ----------
    path_template_directory : str
        Path of the templates directory.
    protrusion : bool, optional
        Generate only templates with protrusion or not. If None, all the
        templates are generated.

    Returns
    -------
    _ : Tuple generator
        cell_mask : np.ndarray, bool
            Binary mask of the cell surface with shape (z, y, x).
        cell_map : np.ndarray, np.float32
            Distance map from cell membrane with shape (z, y, x).
        nuc_mask : np.ndarray, bool
            Binary mask of the nucleus surface with shape (z, y, x).
        nuc_map : np.ndarray, np.float32
            Distance map from nucleus membrane with shape (z, y, x).
        protrusion_mask : np.ndarray, bool
            Binary mask of the protrusion surface with shape (y, x).
        protrusion_map : np.ndarray, np.float32
            Distance map from protrusion region with shape (y, x).

    """
    # check parameters
    stack.check_parameter(
        path_template_directory=str,
        protrusion=(bool, type(None)))

    # get index template
    df = read_index_template(path_template_directory)

    # filter templates with protrusion
    if protrusion is None:
        indices = list(df.loc[:, "id"])
    elif protrusion:
        indices = df.loc[df.loc[:, "protrusion_flag"] == "protrusion", "id"]
    else:
        indices = df.loc[df.loc[:, "protrusion_flag"] == "noprotrusion", "id"]

    # loop over requested templates
    for i in indices:
        template = build_template(
            path_template_directory=path_template_directory,
            i_cell=i,
            index_template=df,
            protrusion=protrusion)

        yield template


def build_template(
        path_template_directory,
        i_cell=None,
        index_template=None,
        protrusion=None):
    """Build template from sparse coordinates. Outcomes are binary masks and
    distance maps, for cell, nucleus and protrusion.

    Parameters
    ----------
    path_template_directory : str
        Path of the templates directory.
    i_cell : int, optional
        Template id to build (between 0 and 317). If None, a random template
        is built.
    index_template : pd.DataFrame, optional
        Dataframe with the templates metadata. If None, dataframe is load from
        'path_template_directory'. Columns are:

        * 'id' instance id.
        * 'shape' shape of the cell image (with the format '{z}_{y}_{x}').
        * 'protrusion_flag' presence or not of protrusion in  the instance.
    protrusion : bool, optional
        Generate only templates with protrusion or not. If None, all the
        templates are generated.

    Returns
    -------
    cell_mask : np.ndarray, bool
        Binary mask of the cell surface with shape (z, y, x).
    cell_map : np.ndarray, np.float32
        Distance map from cell membrane with shape (z, y, x).
    nuc_mask : np.ndarray, bool
        Binary mask of the nucleus surface with shape (z, y, x).
    nuc_map : np.ndarray, np.float32
        Distance map from nucleus membrane with shape (z, y, x).
    protrusion_mask : np.ndarray, bool
        Binary mask of the protrusion surface with shape (y, x).
    protrusion_map : np.ndarray, np.float32
        Distance map from protrusion region with shape (y, x).

    """
    # check parameters
    stack.check_parameter(
        path_template_directory=str,
        i_cell=(int, type(None)),
        index_template=(pd.DataFrame, type(None)),
        protrusion=(bool, type(None)))

    # get index template
    if index_template is None:
        df = read_index_template(path_template_directory)
    else:
        df = index_template

    # filter templates with protrusion
    if protrusion is None:
        indices = list(df.loc[:, "id"])
    elif protrusion:
        indices = df.loc[df.loc[:, "protrusion_flag"] == "protrusion", "id"]
    else:
        indices = df.loc[df.loc[:, "protrusion_flag"] == "noprotrusion", "id"]

    # check specific template or sample one
    if i_cell is not None:
        if i_cell not in indices and protrusion:
            raise ValueError("Requested template {0} does not have protrusion."
                             .format(i_cell))
        elif i_cell not in indices and not protrusion:
            raise ValueError("Requested template {0} has protrusion."
                             .format(i_cell))
    else:
        i_cell = np.random.choice(indices)

    # get metadata and build filename
    shape_str = df.loc[i_cell, "shape"]
    protrusion_flag = df.loc[i_cell, "protrusion_flag"]
    filename = "{0}_{1}_{2}".format(i_cell, shape_str, protrusion_flag)

    # read files
    path = os.path.join(
        path_template_directory, "cell_mask_{0}.npy".format(filename))
    coord_cell_mask = stack.read_array(path)
    path = os.path.join(
        path_template_directory, "cell_map_{0}.npy".format(filename))
    coord_cell_map = stack.read_array(path)
    path = os.path.join(
        path_template_directory, "nuc_mask_{0}.npy".format(filename))
    coord_nuc_mask = stack.read_array(path)
    path = os.path.join(
        path_template_directory, "nuc_map_{0}.npy".format(filename))
    coord_nuc_map = stack.read_array(path)

    # get frame shape
    shape_z, shape_y, shape_x = map(int, shape_str.split("_"))

    # build masks and distance map for cell
    cell_mask = np.zeros((shape_z, shape_y, shape_x), dtype=bool)
    cell_mask[coord_cell_mask[:, 0],
              coord_cell_mask[:, 1],
              coord_cell_mask[:, 2]] = True
    cell_map = np.zeros((shape_z, shape_y, shape_x), dtype=np.float32)
    cell_map[coord_cell_mask[:, 0],
             coord_cell_mask[:, 1],
             coord_cell_mask[:, 2]] = coord_cell_map

    # build masks and distance map for nucleus
    nuc_mask = np.zeros((shape_z, shape_y, shape_x), dtype=bool)
    nuc_mask[coord_nuc_mask[:, 0],
             coord_nuc_mask[:, 1],
             coord_nuc_mask[:, 2]] = True
    nuc_map = np.zeros((shape_z, shape_y, shape_x), dtype=np.float32)
    nuc_map[coord_cell_mask[:, 0],
            coord_cell_mask[:, 1],
            coord_cell_mask[:, 2]] = coord_nuc_map

    # build masks and distance map for protrusion
    if protrusion_flag == "protrusion":
        path = os.path.join(path_template_directory,
                            "protrusion_mask_{0}.npy".format(filename))
        coord_protrusion_mask = stack.read_array(path)
        path = os.path.join(path_template_directory,
                            "protrusion_map_{0}.npy".format(filename))
        coord_protrusion_map = stack.read_array(path)
        protrusion_mask = np.zeros((shape_y, shape_x), dtype=bool)
        protrusion_mask[coord_protrusion_mask[:, 0],
                        coord_protrusion_mask[:, 1]] = True
        cell_mask_2d = stack.maximum_projection(cell_mask.astype(np.uint8))
        cell_mask_2d = cell_mask_2d.astype(bool)
        tuple_coord_cell_mask_2d = np.nonzero(cell_mask_2d)
        coord_cell_mask_2d = np.zeros((cell_mask_2d.sum(), 2), np.uint16)
        coord_cell_mask_2d[:, 0] = tuple_coord_cell_mask_2d[0]
        coord_cell_mask_2d[:, 1] = tuple_coord_cell_mask_2d[1]
        protrusion_map = np.zeros((shape_y, shape_x), dtype=np.float32)
        protrusion_map[coord_cell_mask_2d[:, 0],
                       coord_cell_mask_2d[:, 1]] = coord_protrusion_map
    else:
        protrusion_mask = np.zeros((shape_y, shape_x), dtype=bool)
        protrusion_map = np.zeros((shape_y, shape_x), dtype=np.float32)

    return (cell_mask, cell_map, nuc_mask, nuc_map, protrusion_mask,
            protrusion_map)
