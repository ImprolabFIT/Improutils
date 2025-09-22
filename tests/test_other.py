import unittest

import numpy as np
import numpy.testing as npt

from improutils import midpoint, order_points


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


# -----------------------

if __name__ == "__main__":
    unittest.main()
