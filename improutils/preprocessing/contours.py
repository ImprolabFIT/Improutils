import cv2
import numpy as np


def contour_to_image(contour, img_bin, size=None):
    """Create a new image from the contour.

    Parameters
    ----------
    contour : ndarray
        Contour that represents the area from image to be cropped.
    img_bin : ndarray
        Input binary image.
    size : tuple
        Optional size of the created image.
        If it's not used, the image's size is the same as the
        size of bounding rectangle of the input contour.

    Returns
    -------
    Output cropped image.

    """
    if size is None:
        _, _, w, h = cv2.boundingRect(contour)
        size = (w, h)

    assert type(size) is tuple, "Param size should be a tuple!"
    blank = np.zeros_like(img_bin)
    half_x = int(size[0] * 0.5)
    half_y = int(size[1] * 0.5)

    c = get_center(contour)
    cv2.drawContours(blank, [contour], -1, (255, 255, 255), cv2.FILLED)

    return blank[c[1] - half_y : c[1] + half_y, c[0] - half_x : c[0] + half_x].copy()


def find_contours(img_bin, min_area=0, max_area=np.inf, fill=True, external=True):
    """Find contours in a binary image and filter them by area.

    The function counts the filtered contours and draws them on a new binary
    image. Contours can be optionally filled or just outlined, and either
    external or all contours can be considered.

    Parameters
    ----------
    img_bin : ndarray
        Input binary image.
    min_area : int, optional
        Minimum contour area to keep (smaller contours are discarded). Default is 0.
    max_area : int or float, optional
        Maximum contour area to keep (larger contours are discarded). Default is np.inf.
    fill : bool, optional
        If True, filled contours are drawn; if False, contours are drawn as outlines. Default is True.
    external : bool, optional
        If True, only external contours are considered; if False, all contours are considered. Default is True.

    Returns
    -------
    contour_drawn : ndarray
        Output binary image with drawn filtered contours.
    count : int
        Number of filtered contours.
    contours : list of ndarray
        List of filtered contour arrays.

    """
    mode = cv2.RETR_EXTERNAL
    if not external:
        mode = cv2.RETR_LIST
    contours, _ = cv2.findContours(img_bin, mode, cv2.CHAIN_APPROX_SIMPLE)
    contours = [
        c
        for c in contours
        if cv2.contourArea(c) > min_area and cv2.contourArea(c) < max_area
    ]
    thick = cv2.FILLED
    if not fill:
        thick = 2
    contour_drawn = cv2.drawContours(
        np.zeros(img_bin.shape, dtype=np.uint8),
        contours,
        -1,
        color=(255, 255, 255),
        thickness=thick,
    )
    return contour_drawn, len(contours), contours


def fill_holes(img_bin, close=False, size=5):
    """Fill the holes in found contours. It could merge the contour using close input with appropriate size.

    Parameters
    ----------
    img_bin : ndarray
        Input binary image.
    close : boolean
        If it should merge contours with missing points using close operation.
    size : int
        Size of the close operation element.
    fill : bool, optional
        If True, filled contours are drawn; if False, contours are drawn as outlines. Default is True.

    Returns
    -------
    Output binary image.

    """
    if close:
        struct = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_CLOSE, struct)
    res, _, _ = find_contours(img_bin)
    return res


def find_holes(img_bin, min_area=0, max_area=np.inf, fill=True):
    """Find inner contours (holes) in a binary image and filter them by area.

    The function identifies contours that are inside other contours (holes),
    filters them based on the specified minimum and maximum area, and draws
    them on a new binary image. Contours can be optionally filled or outlined.

    Parameters
    ----------
    img_bin : ndarray
        Input binary image.
    min_area : int, optional
        Minimum contour area to keep (smaller contours are discarded). Default is 0.
    max_area : int or float, optional
        Maximum contour area to keep (larger contours are discarded). Default is np.inf.
    fill : bool, optional
        If True, filled contours are drawn; if False, contours are drawn as outlines. Default is True.

    Returns
    -------
    contours_drawn : ndarray
        Output binary image with drawn filtered hole contours.
    count : int
        Number of filtered hole contours.
    contours : list of ndarray
        List of filtered hole contour arrays.

    """
    contours, hierarchy = cv2.findContours(
        img_bin, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
    )
    # filter out only hole contours (contours that are inside another contour)
    # for more info about hierarchy retrieval modes visit: https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    hole_indices = [i for i in range(len(hierarchy[0])) if hierarchy[0, i, -1] != -1]
    # filter out contours by area
    contours = [
        contours[hole_index]
        for hole_index in hole_indices
        if min_area < cv2.contourArea(contours[hole_index]) <= max_area
    ]
    # draw contours
    thick = cv2.FILLED
    if not fill:
        thick = 2
    contours_drawn = cv2.drawContours(
        np.zeros(img_bin.shape, dtype=np.uint8),
        contours,
        -1,
        color=(255, 255, 255),
        thickness=thick,
    )
    return contours_drawn, len(contours), contours


def get_center(contour):
    """Get the center of a contour in pixels in tuple format.

    Parameters
    ----------
    contour : ndarray
        input contour.

    Returns
    -------
    A tuple with x and y coordinates of the contour's center.

    """
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    return cX, cY
