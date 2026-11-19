import unittest

import cv2
import numpy as np
import numpy.testing as npt

from improutils import (
    convert_pt_from_homogenous,
    convert_pt_to_homogenous,
    convert_pts_from_homogenous,
    convert_pts_to_homogenous,
)


class CoordConversionTestCase(unittest.TestCase):
    def test_pts_to_homogenous_matches_opencv(self):
        pts = np.array([[0.0, 0.0], [1.5, -2.0], [100.0, 250.0]])
        expected = cv2.convertPointsToHomogeneous(pts).reshape(-1, 3)
        npt.assert_allclose(convert_pts_to_homogenous(pts), expected)

    def test_pts_from_homogenous_matches_opencv(self):
        pts = np.array([[2.0, 4.0, 2.0], [-3.0, 1.5, 0.5], [10.0, 20.0, 1.0]])
        expected = cv2.convertPointsFromHomogeneous(pts).reshape(-1, 2)
        npt.assert_allclose(convert_pts_from_homogenous(pts), expected)

    def test_3d_pts_match_opencv(self):
        pts = np.array([[1.0, 2.0, 3.0], [-4.0, 5.0, 6.0]])
        homogenous = convert_pts_to_homogenous(pts)
        npt.assert_allclose(
            homogenous, cv2.convertPointsToHomogeneous(pts).reshape(-1, 4)
        )
        npt.assert_allclose(convert_pts_from_homogenous(homogenous * 2.5), pts)

    def test_single_pt_round_trip(self):
        pt = np.array([3.0, -7.0])
        npt.assert_allclose(convert_pt_to_homogenous(pt), [3.0, -7.0, 1.0])
        self.assertEqual(
            convert_pt_from_homogenous(np.array([6.0, -14.0, 2.0])), (3.0, -7.0)
        )


if __name__ == "__main__":
    unittest.main()
