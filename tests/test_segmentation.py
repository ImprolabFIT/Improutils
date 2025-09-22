import unittest
from pathlib import Path

import cv2
import numpy as np

from improutils import (
    load_image,
    segmentation_one_threshold,
    segmentation_two_thresholds,
    to_3_channels,
    to_gray,
)


class SegmentationCase(unittest.TestCase):
    def test_to_3_channels(self):
        img_path = Path(__name__).parent.absolute() / "tests" / "img" / "test-img.png"
        img_org = load_image(str(img_path))
        img_org = to_gray(img_org)

        img = to_3_channels(img_org)
        self.assertEqual(img.shape[2], 3)

    def test_segmentation_one_threshold(self):
        img = np.zeros((6, 6), np.uint8)
        img[3:5, 2:4] = 255  # 4 white pixels
        img[2:3, 2:4] = 150  # 2 gray pixels

        img_bin1 = segmentation_one_threshold(img, 150)
        img_bin2 = segmentation_one_threshold(img, 100)
        img_bin3 = segmentation_one_threshold(img, -1)

        self.assertEqual(cv2.countNonZero(img_bin1), 4)
        self.assertEqual(cv2.countNonZero(img_bin2), 6)
        self.assertEqual(cv2.countNonZero(img_bin3), img.size)

    def test_segmentation_two_thresholds(self):
        img = np.zeros((6, 6), np.uint8)
        img[3:5, 2:4] = 255  # 4 white pixels
        img[3:5, 4:5] = 30  # 2 dark grey pixels
        img[2:3, 2:4] = 150  # 2 gray pixels

        img_bin1 = segmentation_two_thresholds(img, 0, 30)
        img_bin2 = segmentation_two_thresholds(img, 30, 150)
        img_bin3 = segmentation_two_thresholds(img, 150, 255)
        img_bin4 = segmentation_two_thresholds(img, 0, 150)

        self.assertEqual(cv2.countNonZero(img_bin1), 30)
        self.assertEqual(cv2.countNonZero(img_bin2), 4)
        self.assertEqual(cv2.countNonZero(img_bin3), 6)
        self.assertEqual(cv2.countNonZero(img_bin4), 32)


if __name__ == "__main__":
    unittest.main()
