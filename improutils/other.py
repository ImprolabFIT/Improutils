import numpy as np
import cv2
import os

def midpoint(ptA, ptB):
    """
    Returns the midpoint between two input points.
    """

    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def artificial_circle_image(size):
    """
    Creates image with circles
    Parameters
    ----------
    size : int
        size of the image

    Returns
    -------
    _ : ndarray
        artificial image with circles
    """
    img_art_circ = np.zeros((size, size), dtype=np.uint8)
    step = 10
    for i in range(step, size, step):
        cv2.circle(img_art_circ, (int(size / 2.0), int(size / 2.0)), i - step, np.random.randint(0, 255), thickness=4)
    return img_art_circ


def order_points(pts):
    """
    Sorts the points based on their x-coordinates.
    Parameters
    ----------
    pts : array
        Points to be sorted

    Returns
    -------
    _ : array
        sorted points, the coordinates in top-left, top-right, bottom-right, and bottom-left order
    """

    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (bl, tl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (br, tr) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def pcd_to_depth(pcd, height, width):
    """
    Reduce point-cloud to coordinates, point cloud [x, y, z, rgb] -> depth[x, y, z]

    Parameters
    ----------
    pcd : array
        point cloud
    height : int
        height of captured img
    width : int
        width of a captured img
    Returns
    ----------
    _ : array
        coordinates
    """
    data = pcd
    data = [float(x.split(' ')[2]) for x in data]
    data = np.reshape(data, (height, width))
    return data

def create_file_path(folder, file_name):
    '''Easier defined function to create path for filename inside a folder.

    Parameters
    ----------
    folder : string
        Base folder directory in string notation.
    file_name : string
        File name that should be inside the base folder.
    Returns
    -------
    string
        Path to the newly created file.
    """
    '''
    if not os.path.isdir(folder):
        os.mkdir(folder)

    return os.path.join(folder, file_name)