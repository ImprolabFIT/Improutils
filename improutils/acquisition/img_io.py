import os

import numpy as np
import cv2

def load_image(file_path):
    assert os.path.exists(file_path), 'File does NOT exist! (' + file_path + ')'
    return cv2.imread(file_path)


def save_image(image, file_path):
    return cv2.imwrite(file_path, image)


def copy_to(src, dst, mask):
    '''Python alternative to C++/Java OpenCV's Mat.copyTo().
    More: https://docs.opencv.org/trunk/d3/d63/classcv_1_1Mat.html#a626fe5f96d02525e2604d2ad46dd574f'''
    locs = np.where(mask != 0)  # Get the non-zero mask locations
    dst[locs[0], locs[1]] = src[locs[0], locs[1]]
    return dst
