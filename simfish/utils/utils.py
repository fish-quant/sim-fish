# -*- coding: utf-8 -*-
# Author: Arthur Imbert <arthur.imbert.pro@gmail.com>
# License: BSD 3 clause

"""
Utility functions for simfish.utils subpackage.
"""

import inspect
import warnings

import numpy as np
import pandas as pd

from skimage import img_as_ubyte
from skimage import img_as_float32
from skimage import img_as_float64
from skimage import img_as_uint
from skimage.exposure import rescale_intensity

# TODO add maximum intensity projection


# ### Sanity checks dataframe ###

def check_df(df, features=None, features_without_nan=None):
    """Full safety check of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame or pd.Series
        Dataframe or Series to check.
    features : List[str]
        Names of the expected features.
    features_without_nan : List[str]
        Names of the features to check for the missing values

    Returns
    -------
    _ : bool
        Assert if the dataframe is well formatted.

    """
    # check parameters
    check_parameter(df=(pd.DataFrame, pd.Series),
                    features=(list, type(None)),
                    features_without_nan=(list, type(None)))

    # check features
    if features is not None:
        _check_features_df(df, features)

    # check NaN values
    if features_without_nan is not None:
        _check_features_df(df, features_without_nan)
        _check_nan_df(df, features_without_nan)

    return True


def _check_features_df(df, features):
    """Check that the dataframe contains expected features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check.
    features : List[str]
        Names of the expected features.

    Returns
    -------

    """
    # check columns
    if not set(features).issubset(df.columns):
        raise ValueError("The dataframe does not seem to have the right "
                         "features. {0} instead of {1}"
                         .format(list(df.columns.values), features))

    return


def _check_nan_df(df, features_to_check=None):
    """Check specific columns of the dataframe do not have any missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to check.
    features_to_check : List[str]
        Names of the checked features.

    Returns
    -------

    """
    # count NaN
    nan_count = df.isnull().sum()

    # for the full dataframe...
    if features_to_check is None:
        x = nan_count.sum()
        if x > 0:
            raise ValueError("The dataframe has {0} NaN values.".format(x))

    # ...or for some features
    else:
        nan_count = nan_count[features_to_check]
        x = nan_count.sum()
        if x > 0:
            raise ValueError("The dataframe has {0} NaN values for the "
                             "requested features: \n{1}.".format(x, nan_count))

    return


# ### Sanity checks array ###

def check_array(array, ndim=None, dtype=None, allow_nan=True):
    """Full safety check of an array.

    Parameters
    ----------
    array : np.ndarray
        Array to check.
    ndim : int or List[int]
        Number of dimensions expected.
    dtype : type or List[type]
        Types expected.
    allow_nan : bool
        Allow NaN values or not.

    Returns
    -------
    _ : bool
        Assert if the array is well formatted.

    """
    # check parameters
    check_parameter(array=np.ndarray,
                    ndim=(int, list, type(None)),
                    dtype=(type, list, type(None)),
                    allow_nan=bool)

    # check the dtype
    if dtype is not None:
        _check_dtype_array(array, dtype)

    # check the number of dimension
    if ndim is not None:
        _check_dim_array(array, ndim)

    # check NaN
    if not allow_nan:
        _check_nan_array(array)

    return True


def _check_dtype_array(array, dtype):
    """Check that a np.ndarray has the right dtype.

    Parameters
    ----------
    array : np.ndarray
        Array to check
    dtype : type or List[type]
        Type expected.

    Returns
    -------

    """
    # enlist the dtype expected
    if isinstance(dtype, type):
        dtype = [dtype]

    # check the dtype of the array
    for dtype_expected in dtype:
        if array.dtype == dtype_expected:
            return
    raise TypeError("{0} is not supported yet. Use one of those dtypes "
                    "instead: {1}.".format(array.dtype, dtype))


def _check_dim_array(array, ndim):
    """Check that the array has the right number of dimensions.

    Parameters
    ----------
    array : np.ndarray
        Array to check.
    ndim : int or List[int]
        Number of dimensions expected

    Returns
    -------

    """
    # enlist the number of expected dimensions
    if isinstance(ndim, int):
        ndim = [ndim]

    # check the number of dimensions of the array
    if array.ndim not in ndim:
        raise ValueError("Array can't have {0} dimension(s). Expected "
                         "dimensions are: {1}.".format(array.ndim, ndim))


def _check_nan_array(array):
    """Check that the array does not have missing values.

    Parameters
    ----------
    array : np.ndarray
        Array to check.

    Returns
    -------

    """
    # count nan
    mask = np.isnan(array)
    x = mask.sum()

    # check the NaN values of the array
    if x > 0:
        raise ValueError("Array has {0} NaN values.".format(x))


def check_range_value(array, min_=None, max_=None):
    """Check the support of the array.

    Parameters
    ----------
    array : np.ndarray
        Array to check.
    min_ : int
        Minimum value allowed.
    max_ : int
        Maximum value allowed.

    Returns
    -------
    _ : bool
        Assert if the array has the right range of values.

    """
    # check lowest and highest bounds
    if min_ is not None and array.min() < min_:
        raise ValueError("The array should have a lower bound of {0}, but its "
                         "minimum value is {1}.".format(min_, array.min()))
    if max_ is not None and array.max() > max_:
        raise ValueError("The array should have an upper bound of {0}, but "
                         "its maximum value is {1}.".format(max_, array.max()))

    return True


# ### Sanity checks parameters ###

def check_parameter(**kwargs):
    """Check dtype of the function's parameters.

    Parameters
    ----------
    kwargs : Type or Tuple[Type]
        Map of each parameter with its expected dtype.

    Returns
    -------
    _ : bool
        Assert if the array is well formatted.

    """
    # get the frame and the parameters of the function
    frame = inspect.currentframe().f_back
    _, _, _, values = inspect.getargvalues(frame)

    # compare each parameter with its expected dtype
    for arg in kwargs:
        expected_dtype = kwargs[arg]
        parameter = values[arg]
        if not isinstance(parameter, expected_dtype):
            actual = "'{0}'".format(type(parameter).__name__)
            if isinstance(expected_dtype, tuple):
                target = ["'{0}'".format(x.__name__) for x in expected_dtype]
                target = "(" + ", ".join(target) + ")"
            else:
                target = expected_dtype.__name__
            raise TypeError("Parameter {0} should be a {1}. It is a {2} "
                            "instead.".format(arg, target, actual))

    return True


# ### Image normalization ###

def rescale(tensor, channel_to_stretch=None, stretching_percentile=99.9):
    """Rescale tensor values up to its dtype range (unsigned/signed integers)
    or between 0 and 1 (float).

    Each round and each channel is rescaled independently. Tensor has between
    2 to 5 dimensions, in the following order: (round, channel, z, y, x).

    By default, we rescale the tensor intensity range to its dtype range (or
    between 0 and 1 for float tensor). We can improve the contrast by
    stretching a smaller range of pixel intensity: between the minimum value
    of a channel and percentile value of the channel (cf.
    'stretching_percentile').

    To be consistent with skimage, 64-bit (unsigned) integer images are not
    supported.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to rescale.
    channel_to_stretch : int, List[int] or Tuple[int]
        Channel to stretch. If None, minimum and maximum of each channel are
        used as the intensity range to rescale.
    stretching_percentile : float or int
        Percentile to determine the maximum intensity value used to rescale
        the image. If 1, the maximum pixel intensity is used to rescale the
        image.

    Returns
    -------
    tensor : np.ndarray
        Tensor rescaled.

    """
    # check parameters
    check_parameter(tensor=np.ndarray,
                    channel_to_stretch=(int, list, tuple, type(None)),
                    stretching_percentile=(int, float))
    check_array(tensor,
                ndim=[2, 3, 4, 5],
                dtype=[np.uint8, np.uint16, np.uint32,
                       np.int8, np.int16, np.int32,
                       np.float16, np.float32, np.float64])
    check_range_value(tensor, min_=0)

    # enlist 'channel_to_stretch' if necessary
    if channel_to_stretch is None:
        channel_to_stretch = []
    elif isinstance(channel_to_stretch, int):
        channel_to_stretch = [channel_to_stretch]

    # wrap tensor in 5-d if necessary
    tensor_5d, original_ndim = _wrap_5d(tensor)

    # rescale
    tensor_5d = _rescale_5d(tensor_5d,
                            channel_to_stretch=channel_to_stretch,
                            stretching_percentile=stretching_percentile)

    # rebuild the original tensor shape
    tensor = _unwrap_5d(tensor_5d, original_ndim)

    return tensor


def _wrap_5d(tensor):
    """Increases the number of dimensions of a tensor up to 5.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to wrap.

    Returns
    -------
    tensor_5d : np.ndarray
        Tensor with shape (round, channel, z, y, x).
    original_ndim : int
        Original number of dimensions.

    """
    # wrap tensor in 5-d if necessary
    original_ndim = tensor.ndim
    if original_ndim == 2:
        tensor_5d = tensor[np.newaxis, np.newaxis, np.newaxis, ...]
    elif original_ndim == 3:
        tensor_5d = tensor[np.newaxis, np.newaxis, ...]
    elif original_ndim == 4:
        tensor_5d = tensor[np.newaxis, ...]
    else:
        tensor_5d = tensor

    return tensor_5d, original_ndim


def _unwrap_5d(tensor_5d, original_ndim):
    """Remove useless dimensions from a 5-d tensor.

    Parameters
    ----------
    tensor_5d : np.ndarray
        Tensor with shape (round, channel, z, y, x).
    original_ndim : int
        Original number of dimensions.

    Returns
    -------
    tensor : np.ndarray
        Unwrapped tensor.

    """
    # rebuild the original tensor shape
    if original_ndim == 2:
        tensor = tensor_5d[0, 0, 0, :, :]
    elif original_ndim == 3:
        tensor = tensor_5d[0, 0, :, :, :]
    elif original_ndim == 4:
        tensor = tensor_5d[0, :, :, :, :]
    else:
        tensor = tensor_5d

    return tensor


def _rescale_5d(tensor, channel_to_stretch, stretching_percentile):
    """Rescale tensor values up to its dtype range (unsigned/signed integers)
    or between 0 and 1 (float).

    Each round and each channel is rescaled independently. Tensor has between
    2 to 5 dimensions, in the following order: (round, channel, z, y, x).

    By default, we rescale the tensor intensity range to its dtype range (or
    between 0 and 1 for float tensor). We can improve the contrast by
    stretching a smaller range of pixel intensity: between the minimum value
    of a channel and percentile value of the channel (cf.
    'stretching_percentile').

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to rescale.
    channel_to_stretch : int, List[int] or Tuple[int]
        Channel to stretch. If None, minimum and maximum of each channel are
        used as the intensity range to rescale.
    stretching_percentile : float
        Percentile to determine the maximum intensity value used to rescale
        the image. If 1, the maximum pixel intensity is used to rescale the
        image.

    Returns
    -------
    tensor : np.ndarray
        Tensor rescaled.

    """
    # target intensity range
    target_range = 'dtype'
    if tensor.dtype in [np.float16, np.float32, np.float64]:
        target_range = (0, 1)

    # rescale each round independently
    rounds = []
    for r in range(tensor.shape[0]):

        # rescale each channel independently
        channels = []
        for c in range(tensor.shape[1]):

            # get channel
            channel = tensor[r, c, :, :, :]

            # rescale channel
            if c in channel_to_stretch:
                pa, pb = np.percentile(channel, (0, stretching_percentile))
                channel_rescaled = rescale_intensity(channel,
                                                     in_range=(pa, pb),
                                                     out_range=target_range)
            else:
                channel_rescaled = rescale_intensity(channel,
                                                     out_range=target_range)
            channels.append(channel_rescaled)

        # stack channels
        tensor_4d = np.stack(channels, axis=0)
        rounds.append(tensor_4d)

    # stack rounds
    tensor_5d = np.stack(rounds, axis=0)

    return tensor_5d


def cast_img_uint8(tensor, catch_warning=False):
    """Cast the image in np.uint8 and scale values between 0 and 255.

    Negative values are not allowed as the skimage method 'img_as_ubyte' would
    clip them to 0. Positives values are scaled between 0 and 255, excepted
    if they fit directly in 8 bit (in this case values are not modified).

    Parameters
    ----------
    tensor : np.ndarray
        Image to cast.
    catch_warning : bool
        Catch and ignore UserWarning about possible precision or sign loss.

    Returns
    -------
    tensor : np.ndarray, np.uint8
        Image cast.

    """
    # check tensor dtype
    check_array(tensor,
                ndim=[2, 3, 4, 5],
                dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
                       np.int8, np.int16, np.int32, np.int64,
                       np.float16, np.float32, np.float64])
    if tensor.dtype in [np.float16, np.float32, np.float64]:
        check_range_value(tensor, min_=0, max_=1)
    elif tensor.dtype in [np.int8, np.int16, np.int32, np.int64]:
        check_range_value(tensor, min_=0)

    if tensor.dtype == np.uint8:
        return tensor

    if (tensor.dtype in [np.uint16, np.uint32, np.uint64,
                         np.int16, np.int32, np.int64]
            and tensor.max() <= 255):
        raise ValueError("Tensor values are between {0} and {1}. It fits in 8 "
                         "bits and won't be scaled between 0 and 255. Use "
                         "'tensor.astype(np.uint8)' instead."
                         .format(tensor.min(), tensor.max()))

    # cast tensor
    if catch_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensor = img_as_ubyte(tensor)
    else:
        tensor = img_as_ubyte(tensor)

    return tensor


def cast_img_uint16(tensor, catch_warning=False):
    """Cast the data in np.uint16.

    Negative values are not allowed as the skimage method 'img_as_uint' would
    clip them to 0. Positives values are scaled between 0 and 65535, excepted
    if they fit directly in 16 bit (in this case values are not modified).

    Parameters
    ----------
    tensor : np.ndarray
        Image to cast.
    catch_warning : bool
        Catch and ignore UserWarning about possible precision or sign loss.
    Returns
    -------
    tensor : np.ndarray, np.uint16
        Image cast.

    """
    # check tensor dtype
    check_array(tensor,
                ndim=[2, 3, 4, 5],
                dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
                       np.int8, np.int16, np.int32, np.int64,
                       np.float16, np.float32, np.float64])
    if tensor.dtype in [np.float16, np.float32, np.float64]:
        check_range_value(tensor, min_=0, max_=1)
    elif tensor.dtype in [np.int8, np.int16, np.int32, np.int64]:
        check_range_value(tensor, min_=0)

    if tensor.dtype == np.uint16:
        return tensor

    if (tensor.dtype in [np.uint32, np.uint64, np.int32, np.int64]
            and tensor.max() <= 65535):
        raise ValueError("Tensor values are between {0} and {1}. It fits in "
                         "16 bits and won't be scaled between 0 and 65535. "
                         "Use 'tensor.astype(np.uint16)' instead."
                         .format(tensor.min(), tensor.max()))

    # cast tensor
    if catch_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensor = img_as_uint(tensor)
    else:
        tensor = img_as_uint(tensor)

    return tensor


def cast_img_float32(tensor, catch_warning=False):
    """Cast the data in np.float32.

    If the input data is in (unsigned) integer, the values are scaled between
    0 and 1. When converting from a np.float dtype, values are not modified.

    Parameters
    ----------
    tensor : np.ndarray
        Image to cast.
    catch_warning : bool
        Catch and ignore UserWarning about possible precision or sign loss.

    Returns
    -------
    tensor : np.ndarray, np.float32
        image cast.

    """
    # check tensor dtype
    check_array(tensor,
                ndim=[2, 3, 4, 5],
                dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
                       np.int8, np.int16, np.int32, np.int64,
                       np.float16, np.float32, np.float64])

    # cast tensor
    if catch_warning:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tensor = img_as_float32(tensor)
    else:
        tensor = img_as_float32(tensor)

    return tensor


def cast_img_float64(tensor):
    """Cast the data in np.float64.

    If the input data is in (unsigned) integer, the values are scaled between
    0 and 1. When converting from a np.float dtype, values are not modified.

    Parameters
    ----------
    tensor : np.ndarray
        Tensor to cast.

    Returns
    -------
    tensor : np.ndarray, np.float64
        Tensor cast.

    """
    # check tensor dtype
    check_array(tensor,
                ndim=[2, 3, 4, 5],
                dtype=[np.uint8, np.uint16, np.uint32, np.uint64,
                       np.int8, np.int16, np.int32, np.int64,
                       np.float16, np.float32, np.float64])

    # cast tensor
    tensor = img_as_float64(tensor)

    return tensor
