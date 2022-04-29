import numpy as np
import cv2

# Dimensionless descriptors
from improutils import find_contours


class ShapeDescriptors:
    def form_factor(area, perimeter):
        return (4 * np.pi * area) / (perimeter * perimeter)

    def roundness(area, max_diameter):
        return (4 * area) / (np.pi * max_diameter * max_diameter)

    def aspect_ratio(min_diameter, max_diameter):
        return min_diameter / max_diameter;

    def convexity(perimeter, convex_perimeter):
        return convex_perimeter / perimeter

    def solidity(area, convex_area):
        return area / convex_area

    def compactness(area, max_diameter):
        return np.sqrt(4 / np.pi * area) / max_diameter;

    def extent(area, bounding_rectangle_area):
        return area / bounding_rectangle_area;

"""
An internal, helper function.
Shall not be called directly by the user.
Given a binary image, finds contours, and returns the result.
If no contours were found, throws an exception.

Parameters
----------
bin_im : ndarray
    binary image. This image contains only black and white values.
    Traditionally, you get it from the segmentation process.

Returns
-------
_ : number
    The array with contours. The return type is specified by
    the find_contours() function.
Throws
-------
_ : a ValueError exception if any of the following contitions hold:
    [+] No contour was found in the input image.
    This traditionally happens, if there are no white pixels in the input image
"""
def _getContoursAndValidate(bin_im):
    _, _, conts = find_contours(bin_im)
    if(len(conts) == 0):
        raise ValueError("No contours were found");
    return conts;

"""
Aka Špičatost.
Allows to determine the form factor of a contour, which
will be calculated from the input binary image
Parameters
----------
bin_im : ndarray
    binary image. This image contains only black and white values.
    Traditionally, you get it from the segmentation process.

Returns
-------
_ : number
    The number, describing the contour property
Throws
-------
_ : a ValueError exception if any of the following contitions hold:
    [+] No contour was found in the input image.
    This traditionally happens, if there are no white pixels in the input image
"""
def form_factor(bin_im):
    conts = _getContoursAndValidate(bin_im)
    return ShapeDescriptors.form_factor(cv2.contourArea(conts[0]), cv2.arcLength(conts[0], True))

"""
Allows to determine the roundness of a contour, which
will be calculated from the input binary image
Parameters
----------
bin_im : ndarray
    binary image. This image contains only black and white values.
    Traditionally, you get it from the segmentation process.

Returns
-------
_ : number
    The number, describing the contour property
Throws
-------
_ : a ValueError exception if any of the following contitions hold:
    [+] No contour was found in the input image.
    This traditionally happens, if there are no white pixels in the input image
"""
def roundness(bin_im):
    conts = _getContoursAndValidate(bin_im)
    area = cv2.contourArea(conts[0])
    _, radius = cv2.minEnclosingCircle(conts[0])
    r = ShapeDescriptors.roundness(area, 2 * radius)
    if r > 1: r = 1
    return r

"""
Allows to determine the aspect ratio of a contour, which
will be calculated from the input binary image
Parameters
----------
bin_im : ndarray
    binary image. This image contains only black and white values.
    Traditionally, you get it from the segmentation process.

Returns
-------
_ : number
    The number, describing the contour property
Throws
-------
_ : a ValueError exception if any of the following contitions hold:
    [+] No contour was found in the input image.
    This traditionally happens, if there are no white pixels in the input image
"""
def aspect_ratio(bin_im):
    conts = _getContoursAndValidate(bin_im)
    dims = cv2.minAreaRect(conts[0])[1]
    min_diameter = min(dims)
    max_diameter = max(dims)
    return ShapeDescriptors.aspect_ratio(min_diameter, max_diameter)


"""
Allows to determine the convexity of a contour, which
will be calculated from the input binary image
Parameters
----------
bin_im : ndarray
    binary image. This image contains only black and white values.
    Traditionally, you get it from the segmentation process.

Returns
-------
_ : number
    The number, describing the contour property
Throws
-------
_ : a ValueError exception if any of the following contitions hold:
    [+] No contour was found in the input image.
    This traditionally happens, if there are no white pixels in the input image
"""
def convexity(bin_im):
    conts = _getContoursAndValidate(bin_im)
    hull = cv2.convexHull(conts[0], None, True, True)
    per = cv2.arcLength(conts[0], True)
    conv_per = cv2.arcLength(hull, True)
    r = ShapeDescriptors.convexity(per, conv_per)
    if r > 1: r = 1
    return r


"""
Allows to determine the solidity of a contour, which
will be calculated from the input binary image
Parameters
----------
bin_im : ndarray
    binary image. This image contains only black and white values.
    Traditionally, you get it from the segmentation process.

Returns
-------
_ : number
    The number, describing the contour property
Throws
-------
_ : a ValueError exception if any of the following contitions hold:
    [+] No contour was found in the input image.
    This traditionally happens, if there are no white pixels in the input image
"""
def solidity(bin_im):
    conts = _getContoursAndValidate(bin_im)
    hull = cv2.convexHull(conts[0], None, True, True)
    area = cv2.contourArea(conts[0])
    conv_area = cv2.contourArea(hull)
    r = ShapeDescriptors.solidity(area, conv_area)
    if r > 1: r = 1
    return r


"""
Allows to determine the compactness of a contour, which
will be calculated from the input binary image
Parameters
----------
bin_im : ndarray
    binary image. This image contains only black and white values.
    Traditionally, you get it from the segmentation process.

Returns
-------
_ : number
    The number, describing the contour property
Throws
-------
_ : a ValueError exception if any of the following contitions hold:
    [+] No contour was found in the input image.
    This traditionally happens, if there are no white pixels in the input image
"""
def compactness(bin_im):
    conts = _getContoursAndValidate(bin_im)
    area = cv2.contourArea(conts[0])
    max_diameter = max(cv2.minAreaRect(conts[0])[1])
    r = ShapeDescriptors.compactness(area, max_diameter)
    if r > 1: r = 1
    return r


"""
Allows to determine the extent of a contour, which
will be calculated from the input binary image
Parameters
----------
bin_im : ndarray
    binary image. This image contains only black and white values.
    Traditionally, you get it from the segmentation process.

Returns
-------
_ : number
    The number, describing the contour property
Throws
-------
_ : a ValueError exception if any of the following contitions hold:
    [+] No contour was found in the input image.
    This traditionally happens, if there are no white pixels in the input image
"""
def extent(bin_im):
    conts = _getContoursAndValidate(bin_im)
    area = cv2.contourArea(conts[0])
    w, h = cv2.minAreaRect(conts[0])[1]
    return ShapeDescriptors.extent(area, w * h)
