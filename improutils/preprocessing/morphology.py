import cv2
import numpy as np

def erode(img, kernel_size=3, iterations=1, kernel=None):
    """
    Erodes an image using the specified kernel.
    Parameters
    ----------
    - img: numpy.ndarray
        The input image to be eroded.
    - kernel_size: int, optional
        The size of the kernel used for erosion. Default is 3.
    - iterations: int, optional
        The number of times erosion is applied. Default is 1.
    - kernel: numpy.ndarray, optional
        The custom kernel to be used for erosion. If not provided, a square kernel of size `kernel_size` will be used.
    Returns
    -------
    The eroded image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if not kernel else kernel
    return cv2.erode(img, kernel=kernel, iterations=iterations)


def dilate(img, kernel_size=3, iterations=1, kernel=None):
    """
    Dilates an image using the specified kernel.
    Parameters
    ----------
    - img: numpy.ndarray
        The input image to be dilated.
    - kernel_size: int, optional
        The size of the kernel used for dilation. Default is 3.
    - iterations: int, optional
        The number of times the dilation is applied. Default is 1.
    - kernel: numpy.ndarray, optional
        The custom kernel to be used for dilation. If not provided, a square kernel of size `kernel_size` will be used.
    Returns
    -------
    The dilated image.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if not kernel else kernel
    return cv2.dilate(img, kernel=kernel, iterations=iterations)


def open(img, kernel_size=3, iterations=1, kernel=None):
    """
    Applies the morphological opening operation on the input image.
    Parameters
    ----------
    - img: numpy.ndarray
        The input image.
    - kernel_size: int, optional
        The size of the kernel used for the opening operation. Default is 3.
    - iterations: int, optional
        The number of times the opening operation is applied. Default is 1.
    - kernel: numpy.ndarray, optional
        The custom kernel used for the opening operation. If not provided, a square kernel of size `kernel_size` will be used.
    Returns
    -------
    The image after applying the morphological opening operation.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if not kernel else kernel
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)


def close(img, kernel_size=3, iterations=1, kernel=None):
    """
    Applies morphological closing operation on the input image.
    Parameters
    ----------
    - img: numpy.ndarray
        The input image.
    - kernel_size: int, optional
        The size of the kernel used for the closing operation. Default is 3.
    - iterations: int, optional
        The number of times the closing operation is applied. Default is 1.
    - kernel: numpy.ndarray, optional
        The custom kernel used for the closing operation. If not provided, a square kernel of size `kernel_size` will be used.
    Returns
    -------
    The image after applying the closing operation.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if not kernel else kernel
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)


def morphological_gradient(img, kernel_size=3, iterations=1, kernel=None):
    """
    Applies morphological gradient operation on the input image.
    Parameters
    ----------
    - img: numpy.ndarray
        The input image.
    - kernel_size: int, optional
        The size of the kernel used for morphological operations. Defaults to 3.
    - iterations: int, optional
        The number of times the morphological operation is applied. Defaults to 1.
    - kernel: numpy.ndarray, optional
        The custom kernel used for morphological operations. If not provided, a square kernel of size `kernel_size` will be used.
    Returns
    -------
    The image after applying the morphological gradient operation.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if not kernel else kernel
    return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel, iterations=iterations)


def top_hat(img, kernel_size=3, iterations=1, kernel=None):
    """
    Applies the top hat morphological operation on the input image.
    Parameters
    ----------
    - img: numpy.ndarray
        The input image.
    - kernel_size: int, optional
        The size of the kernel used for the operation. Default is 3.
    - iterations: int, optional
        The number of times the operation is applied. Default is 1.
    - kernel: numpy.ndarray, optional
        The custom kernel to be used for the operation. If not provided, a square kernel of size `kernel_size` will be used.
    Returns
    -------
    The image after applying the top hat operation.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8) if not kernel else kernel
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel, iterations=iterations)


def black_hat(img, kernel_size=3, iterations=1, kernel=None):
    """
    Applies the black hat morphological operation on the input image.
    Parameters
    ----------
    - img: numpy.ndarray
        The input image.
    - kernel_size: int, optional
        The size of the kernel used for the operation. Default is 3.
    - iterations: int, optional
        The number of times the operation is applied. Default is 1.
    - kernel: numpy.ndarray, optional
        The kernel used for the operation. If not provided, a square kernel of size `kernel_size` will be used.
    Returns
    -------
    The image after applying the black hat operation.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel, iterations=iterations)
