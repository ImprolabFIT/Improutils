import cv2
import numpy as np

# Dimensionless descriptors


class ShapeDescriptors:

    """An internal class for computing shape descriptors. Not to be used by the programmer."""

    def form_factor(area: float, perimeter: float) -> float:
        """Compute the form factor of a shape based on area and perimeter."""
        return (4 * np.pi * area) / (perimeter * perimeter)

    def roundness(area: float, max_diameter: float) -> float:
        """Compute the roundness of a shape based on area and maximum diameter."""
        return (4 * area) / (np.pi * max_diameter * max_diameter)

    def aspect_ratio(min_diameter: float, max_diameter: float) -> float:
        """Compute the aspect ratio of a shape as the ratio of minimum to maximum diameter."""
        return min_diameter / max_diameter

    def convexity(perimeter: float, convex_perimeter: float) -> float:
        """Compute the convexity of a shape as the ratio of convex perimeter to actual perimeter."""
        return convex_perimeter / perimeter

    def solidity(area: float, convex_area: float) -> float:
        """Compute the solidity of a shape as the ratio of area to its convex hull area."""
        return area / convex_area

    def compactness(area: float, max_diameter: float) -> float:
        """Compute the compactness of a shape based on area and maximum diameter."""
        return np.sqrt(4 / np.pi * area) / max_diameter

    def extent(area: float, bounding_rectangle_area: float) -> float:
        """Compute the extent of a shape as the ratio of area to bounding rectangle area."""
        return area / bounding_rectangle_area


def form_factor(contour: np.ndarray) -> float:
    """Determine the contour's form factor.

    Aka "špičatost".

    Parameters
    ----------
    contour : ndarray
        The contour for the calculation.

    Returns
    -------
    The number, describing the contour's property

    """
    return ShapeDescriptors.form_factor(
        cv2.contourArea(contour), cv2.arcLength(contour, True)
    )


def roundness(contour: np.ndarray) -> float:
    """Determine the contour's roundness.

    Aka "kulatost".

    Parameters
    ----------
    contour : ndarray
        The contour for the calculation.

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


def aspect_ratio(contour: np.ndarray) -> float:
    """Determine the contour's aspect ratio.

    Aka "poměr stran".

    Parameters
    ----------
    contour : ndarray
        The contour for the calculation.

    Returns
    -------
    The number, describing the contour's property

    """
    dims = cv2.minAreaRect(contour)[1]
    min_diameter = min(dims)
    max_diameter = max(dims)
    return ShapeDescriptors.aspect_ratio(min_diameter, max_diameter)


def convexity(contour: np.ndarray) -> float:
    """Determine the contour's convexity.

    Aka "konvexita, vypouklost".

    Parameters
    ----------
    contour : ndarray
        The contour for the calculation.

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


def solidity(contour: np.ndarray) -> float:
    """Determine the contour's solidity.

    Aka "plnost, celistvost".

    Parameters
    ----------
    contour : ndarray
        The contour for the calculation.

    Returns
    -------
    The number, describing the contour's property

    """
    hull = cv2.convexHull(contour, None, True, True)
    area = cv2.contourArea(contour)
    conv_area = cv2.contourArea(hull)
    r = ShapeDescriptors.solidity(area, conv_area)
    if r > 1:
        r = 1
    return r


def compactness(contour: np.ndarray) -> float:
    """Determine the contour's compactness.

    Aka "kompaktnost, hutnost".

    Parameters
    ----------
    contour : ndarray
        The contour for the calculation.

    Returns
    -------
    The number, describing the contour's property

    """
    area = cv2.contourArea(contour)
    max_diameter = max(cv2.minAreaRect(contour)[1])
    r = ShapeDescriptors.compactness(area, max_diameter)
    if r > 1:
        r = 1
    return r


def extent(contour: np.ndarray) -> float:
    """Determine the contour's extent.

    Aka "dosah, rozměrnost".

    Parameters
    ----------
    contour : ndarray
        The contour for the calculation.

    Returns
    -------
    The number, describing the contour's property

    """
    area = cv2.contourArea(contour)
    w, h = cv2.minAreaRect(contour)[1]
    return ShapeDescriptors.extent(area, w * h)
