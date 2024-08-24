import numpy as np
import cv2

# Dimensionless descriptors
from improutils import find_contours


class ShapeDescriptors:
    """
    An internal class for computing shape descriptors.
    Not to be used by the programmer.
    """
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

def form_factor(contour):
    """
    Aka "špičatost".
    Allows to determine the contour's form factor.

    Parameters
    ----------
    contour : ndarray
    Returns
    -------
    The number, describing the contour's property

    """

    return ShapeDescriptors.form_factor(cv2.contourArea(contour), cv2.arcLength(contour, True))


def roundness(contour):
    """
    Aka "kulatost".
    Allows to determine the contour's roundness.
    Parameters
    ----------
    contour : ndarray
    Returns
    -------
    The number, describing the contour's property
    """

    area = cv2.contourArea(contour)
    _, radius = cv2.minEnclosingCircle(contour)
    r = ShapeDescriptors.roundness(area, 2 * radius)
    if r > 1:
        r = 1
    return r


def aspect_ratio(contour):
    """
    Aka "poměr stran".
    Allows to determine the contour's aspect ratio.

    Parameters
    ----------
    contour : ndarray
    Returns
    -------
    The number, describing the contour's property
    """

    dims = cv2.minAreaRect(contour)[1]
    min_diameter = min(dims)
    max_diameter = max(dims)
    return ShapeDescriptors.aspect_ratio(min_diameter, max_diameter)


def convexity(contour):
    """
    Aka "konvexita, vypouklost".
    Allows to determine the contour's convexity.
    Parameters
    ----------
    contour : ndarray
    Returns
    -------
    The number, describing the contour's property
    """

    hull = cv2.convexHull(contour, None, True, True)
    per = cv2.arcLength(contour, True)
    conv_per = cv2.arcLength(hull, True)
    r = ShapeDescriptors.convexity(per, conv_per)
    if r > 1:
        r = 1
    return r


def solidity(contour):
    """
    Aka "plnost, celistvost".
    Allows to determine the contour's solidity.
    Parameters
    ----------
    contour : ndarray
    Returns
    -------
    The number, describing the contour's property
    """

    hull = cv2.convexHull(contour, None, True, True)
    area = cv2.contourArea(contour)
    conv_area = cv2.contourArea(hull)
    r = ShapeDescriptors.solidity(area, conv_area)
    if r > 1: r = 1
    return r


def compactness(contour):
    """
    Aka "kompaktnost, hutnost".
    Allows to determine the contour's compactness.
    Parameters
    ----------
    contour : ndarray
    Returns
    -------
    The number, describing the contour's property
    """

    area = cv2.contourArea(contour)
    max_diameter = max(cv2.minAreaRect(contour)[1])
    r = ShapeDescriptors.compactness(area, max_diameter)
    if r > 1: r = 1
    return r


def extent(contour):
    """
    Aka "dosah, rozměrnost".
    Allows to determine the contour's extent.
    Parameters
    ----------
    contour : ndarray
    Returns
    -------
    The number, describing the contour's property
    """

    area = cv2.contourArea(contour)
    w, h = cv2.minAreaRect(contour)[1]
    return ShapeDescriptors.extent(area, w * h)
