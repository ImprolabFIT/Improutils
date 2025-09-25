import unittest
from pathlib import Path

import cv2
import numpy as np

from improutils import (
    contour_to_image,
    crop,
    crop_by_bounding_rect,
    fill_holes,
    find_contours,
    get_center,
    load_image,
    resize,
    rotate,
    segmentation_two_thresholds,
    to_gray,
    to_hsv,
)

base_path = Path(__name__).parent.absolute() / "tests" / "img"


class PreprocessingTestCase(unittest.TestCase):
    def test_to_gray(self):
        img_org = load_image(str(base_path / "test-img.png"))
        img = to_gray(img_org)

        self.assertEqual(len(img.shape), 2)

    def test_to_hsv(self):
        blue = ((90, 0, 0), (135, 255, 255))
        img_org = np.zeros((10, 10, 3), np.uint8)
        img_org[:, 2:5] = (255, 0, 0)  # blue

        img = to_hsv(img_org)
        img_bin = cv2.inRange(img, blue[0], blue[1])

        self.assertEqual(img_org.shape, img.shape)
        self.assertEqual(cv2.countNonZero(img_bin), (5 - 2) * 10)

    def test_resize(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img = resize(img, (10, 5))

        self.assertEqual(img.shape, (5, 10, 3))

    def test_crop(self):
        img = np.zeros((10, 10, 3), np.uint8)
        img_crop_org = crop(img, 0, 0, 10, 10)

        tl_x = 2
        tl_y = 2
        br_x = 3
        br_y = 3
        img_crop = crop(img, tl_x, tl_y, br_x, br_y)

        self.assertEqual(img.shape, img_crop_org.shape)
        self.assertEqual(img_crop.shape, (br_x - tl_x, br_y - tl_y, 3))

    def test_crop_by_bounding_rect(self):
        img_org = load_image(str(base_path / "test-img.png"))
        blue = ((90, 0, 0), (135, 255, 255))

        img = to_hsv(img_org)
        img_bin = cv2.inRange(img, blue[0], blue[1])

        img_crop = crop_by_bounding_rect(img_bin)
        c_shape = (231, 239)
        self.assertEqual(img_crop.shape, c_shape)

    def test_rotate(self):
        img = np.zeros((4, 4, 3), np.uint8)
        img[1, 1] = 255

        img_rot0 = rotate(img, 0)
        img_rot360 = rotate(img, 360)
        img_rot90 = rotate(img, 90)

        self.assertFalse(np.bitwise_xor(img, img_rot0).any())
        self.assertFalse(np.bitwise_xor(img, img_rot360).any())
        self.assertFalse(np.bitwise_xor(img[1, 1], img_rot90[3, 1]).any())  # ?


class ContoursTestCase(unittest.TestCase):
    def test_contour_to_image(self):
        size = (6, 6)
        img = np.zeros(size, np.uint8)
        img[2:4, 2:4] = 255
        img_bin = segmentation_two_thresholds(img, 250, 255)
        _, _, contours = find_contours(img_bin)

        img_contour = contour_to_image(contours[0], img, (size[0] * 3, size[1] * 3))

        self.assertFalse(np.bitwise_xor(img, img_contour).any())

    def test_find_contours(self):
        img = load_image(str(base_path / "test-img.png"))
        img = to_gray(img)
        img_bin = segmentation_two_thresholds(img, 0, 250)

        contour_drawn, count, contours = find_contours(img_bin)
        self.assertEqual(count, 4)

    def test_fill_holes(self):
        eps = 100
        img = load_image(str(base_path / "test-img.png"))
        img = to_gray(img)
        img_bin = segmentation_two_thresholds(img, 0, 250)

        contour_drawn_f, _, _ = find_contours(img_bin)
        contour_drawn, _, _ = find_contours(img_bin, fill=False)

        img_bin_fill = fill_holes(contour_drawn)

        self.assertLess(cv2.norm(contour_drawn_f - img_bin_fill), eps)

    def test_get_center(self):
        ref_center = (290, 519)
        img = load_image(str(base_path / "test-img.png"))
        img = to_gray(img)
        img_bin = segmentation_two_thresholds(img, 70, 100)

        contour_drawn, _, contours = find_contours(img_bin, fill=False)
        center = get_center(contours[0])

        self.assertEqual(center, ref_center)


if __name__ == "__main__":
    unittest.main()
