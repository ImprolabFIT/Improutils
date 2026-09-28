import unittest

import cv2
import numpy as np
import numpy.testing as npt

from improutils import midpoint, order_points
from improutils.other import _radial_factor, _tangential_shift


class OtherCase(unittest.TestCase):
    def test_midpoint(self):
        # -----------------------
        # test for array input
        m = midpoint([0, 0], [10, 10])
        self.assertEqual(m, (5.0, 5.0))
        # -----------------------
        # test for tuple input
        m = midpoint((0, 0), (10, 10))
        self.assertEqual(m, (5.0, 5.0))
        # -----------------------
        # test for np array input
        m = midpoint(np.array([0, 0]), np.array([10, 10]))
        self.assertEqual(m, (5.0, 5.0))

    def test_order_points(self):
        # -----------------------
        # classic test
        pts = np.array([[0, 5], [1, 5], [0, 4], [1, 4]])
        e = order_points(pts)
        npt.assert_array_equal(e, [[0, 5], [1, 5], [1, 4], [0, 4]])
        # -----------------------
        # edge cases, that require an exception thrown
        thrown = False
        # edge test: put in array (not ndarray)
        try:
            thrown = False
            e = order_points([[0, 5], [1, 5], [0, 4], [1, 4]])
        except ValueError:
            thrown = True

        if not thrown:
            print(
                "An error was not thrown, expected the tested function to throw an error for the input data provided."
            )
            self.assertEqual(1, 0)
        # -----------------------
        # edge test: put in tuple (not ndarray)
        try:
            thrown = False
            e = order_points(([0, 5], [1, 5], [0, 4], [1, 4]))
        except ValueError:
            thrown = True

        if not thrown:
            print(
                "An error was not thrown, expected the tested function to throw an error for th einput data provided."
            )
            self.assertEqual(1, 0)
        # -----------------------
        # edge test: put in ndarray
        input_ = np.array([[0, 5], [1, 5], [0, 4]])
        try:
            thrown = False
            e = order_points(input_)
        except ValueError:
            thrown = True

        if not thrown:
            print(
                "An error was not thrown, expected the tested function to throw an error for th einput data provided."
            )
            self.assertEqual(1, 0)
        # -----------------------
        # edge test: put in ndarray, but only 2 points (less than 4)
        input_ = np.array([[0, 5], [1, 5]])
        try:
            thrown = False
            e = order_points(input_)
        except ValueError:
            thrown = True

        if not thrown:
            print(
                "An error was not thrown, expected the tested function to throw an error for the input data provided."
            )
            self.assertEqual(1, 0)
        # -----------------------
        # edge test: put in ndarray, but only 1 point (less than 4)
        input_ = np.array([[0, 5]])
        try:
            thrown = False
            e = order_points(input_)
        except ValueError:
            thrown = True

        if not thrown:
            print(
                "An error was not thrown, expected the tested function to throw an error for the input data provided."
            )
            self.assertEqual(1, 0)
        # -----------------------
        # edge test: put in empty ndarray (less than 4 points)
        input_ = np.array([])
        try:
            thrown = False
            e = order_points(input_)
        except ValueError:
            thrown = True

        if not thrown:
            print(
                "An error was not thrown, expected the tested function to throw an error for the input data provided."
            )
            self.assertEqual(1, 0)


    def test_distortion_matches_opencv(self):
        # the teaching implementation of the distortion model must match OpenCV,
        # points (x, y, 1) projected with identity pose and camera matrix
        # are only distorted
        xv, yv = np.meshgrid(np.linspace(-0.1, 0.1, 20), np.linspace(-0.1, 0.1, 20))
        points = np.dstack([xv, yv, np.ones_like(xv)]).reshape(-1, 1, 3)

        for k, p in [
            ((0.3, -0.2, 0.1, 0.05, 0.02, 0.01), (0.01, -0.02)),
            ((-5.0, 20.0, -3.0, 1.0, 0.5, 0.2), (0.05, 0.03)),
            ((0.0,) * 6, (0.0, 0.0)),
        ]:
            radial = _radial_factor(xv, yv, k)
            dx, dy = _tangential_shift(xv, yv, p)

            # OpenCV order of distortion coefficients: k1, k2, p1, p2, k3, k4, k5, k6
            dist_coeffs = np.array([k[0], k[1], p[0], p[1], k[2], k[3], k[4], k[5]])
            expected, _ = cv2.projectPoints(
                points, np.zeros(3), np.zeros(3), np.eye(3), dist_coeffs
            )
            expected = expected.reshape(20, 20, 2)

            npt.assert_allclose(xv * radial + dx, expected[..., 0], atol=1e-12)
            npt.assert_allclose(yv * radial + dy, expected[..., 1], atol=1e-12)


# -----------------------

if __name__ == "__main__":
    unittest.main()
