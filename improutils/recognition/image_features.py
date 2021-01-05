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


# Špičatost
def form_factor(bin_im):
    _, _, conts = find_contours(bin_im)
    return ShapeDescriptors.form_factor(cv2.contourArea(conts[0]), cv2.arcLength(conts[0], True))


# Kulatost
def roundness(bin_im):
    _, _, conts = find_contours(bin_im)
    area = cv2.contourArea(conts[0])
    _, radius = cv2.minEnclosingCircle(conts[0])
    r = ShapeDescriptors.roundness(area, 2 * radius)
    if r > 1: r = 1
    return r


# Poměr stran
def aspect_ratio(bin_im):
    _, _, conts = find_contours(bin_im)
    dims = cv2.minAreaRect(conts[0])[1]
    min_diameter = min(dims)
    max_diameter = max(dims)
    return ShapeDescriptors.aspect_ratio(min_diameter, max_diameter)


# Konvexita, vypouklost
def convexity(bin_im):
    _, _, conts = find_contours(bin_im)
    hull = cv2.convexHull(conts[0], None, True, True)
    per = cv2.arcLength(conts[0], True)
    conv_per = cv2.arcLength(hull, True)
    r = ShapeDescriptors.convexity(per, conv_per)
    if r > 1: r = 1
    return r


# Plnost, celistvost
def solidity(bin_im):
    _, _, conts = find_contours(bin_im)
    hull = cv2.convexHull(conts[0], None, True, True)
    area = cv2.contourArea(conts[0])
    conv_area = cv2.contourArea(hull)
    r = ShapeDescriptors.solidity(area, conv_area)
    if r > 1: r = 1
    return r


# Kompaktnost, hutnost
def compactness(bin_im):
    _, _, conts = find_contours(bin_im)
    area = cv2.contourArea(conts[0])
    max_diameter = max(cv2.minAreaRect(conts[0])[1])
    r = ShapeDescriptors.compactness(area, max_diameter)
    if r > 1: r = 1
    return r


# Dosah, rozměrnost
def extent(bin_im):
    _, _, conts = find_contours(bin_im)
    area = cv2.contourArea(conts[0])
    w, h = cv2.minAreaRect(conts[0])[1]
    return ShapeDescriptors.extent(area, w * h)