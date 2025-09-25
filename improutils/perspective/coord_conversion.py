import numpy as np


# Functions for conversion within different types of coordinates
def convert_pt_to_homogenous(pt):
    """Convert the input point from inhomogeneous coordinates to homogeneous ones.

    Parameters
    ----------
    pt : ndarray
        Input point in inhomogeneous coordinates.

    Returns
    -------
    _ : ndarray
        Input point in homogeneous coordinates.

    """
    return np.append(pt, np.array(1))


def convert_pt_from_homogenous(pt):
    """Convert input point in homogeneous coordinates to inhomogeneous.

    Parameters
    ----------
    pt : ndarray
        Input point in homogeneous coordinates.

    Returns
    -------
    _ : tuple
        Input point in inhomogeneous coordinates.

    """
    return tuple([elem / pt[-1] for elem in pt[:-1]])


def convert_pts_to_homogenous(pts):
    """Convert input points from inhomogeneous to homogeneous coordinates.

    Parameters
    ----------
    pts : array_like, shape (n_points, n_dims)
        Inhomogeneous input points.

    Returns
    -------
    homogeneous_pts : ndarray, shape (n_points, n_dims + 1)
        Points in homogeneous coordinates, with a 1 appended as the last component.

    """
    return np.array([convert_pt_to_homogenous(pt) for pt in pts])


def convert_pts_from_homogenous(pts):
    """Convert input points in homogeneous coordinates to inhomogeneous.

    Parameters
    ----------
    pts : array_like, shape (n_points, n_dims + 1) in homogeneous coordinates.
        Input points in homogeneous coordinates.

    Returns
    -------
    inhomogeneous_pts : ndarray, shape (n_points, n_dims)
        Points in inhomogeneous coordinates.

    """
    return np.array([convert_pt_from_homogenous(pt) for pt in pts])
