import os

import numpy as np
import cv2


def load_image(file_path):
    """
    @brief Loads an image from a file.

    The function calls cv2.imread() to load image from the specified file
    and then return it. If the image cannot be read, the AssertError exception is thrown.

    For more info about formats, see cv2.imread() documentation

    Parameters
    ----------
    file_path : string
        A path to an image file
    Returns
    -------
    Loaded image in numpy.ndarray
    """
    assert os.path.exists(file_path), 'File does NOT exist! (' + file_path + ')'
    return cv2.imread(file_path)


def save_image(image, file_path):
    """
    @brief Save an image to a file.

    The function calls cv2.imwrite() to save an image to the specified file. The image format is chosen based on the
    filename extension.

    Parameters
    ----------
    image : numpy.ndarray
        Pixel values
    file_path : string
        A path to an image file
    Returns
    -------
    True if image is saved successfully
    """
    return cv2.imwrite(file_path, image)


def copy_to(src, dst, mask):
    """
    @brief Copies source image pixel to destination image using mask matrix.

    This function is Python alternative to C++/Java OpenCV's Mat.copyTo().
    More: https://docs.opencv.org/trunk/d3/d63/classcv_1_1Mat.html#a626fe5f96d02525e2604d2ad46dd574f

    Parameters
    ----------
    src : numpy.ndarray
        Source image
    dst : numpy.ndarray
        Destination image
    mask : numpy.ndarray
        Binary image that specifies which pixels are copied. Value 1 means true
    Returns
    -------
    Destination image with copied pixels from source image
    """
    locs = np.where(mask != 0)  # Get the non-zero mask locations
    dst[locs[0], locs[1]] = src[locs[0], locs[1]]
    return dst
