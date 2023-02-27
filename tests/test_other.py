import unittest
from pathlib import Path

from improutils.other import *
import numpy.testing as npt

class OtherCase(unittest.TestCase):

    def test_midpoint(self):
        # -----------------------
        # test for array input
        m = midpoint([0, 0], [10, 10]);
        self.assertEqual(m, (5.0, 5.0));

        # -----------------------
        # test for tuple input
        m = midpoint((0, 0), (10, 10));
        self.assertEqual(m, (5.0, 5.0));

        # -----------------------
        # test for np array input
        m = midpoint(np.array([0, 0]), np.array([10, 10]));
        self.assertEqual(m, (5.0, 5.0));


        # -----------------------
        thrown = False;
        # edge cases, that require exception throw
        # 3D point input
        try:
            thrown = False;
            midpoint(np.array([0, 0, 0]), np.array([10, 10, 10]));
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);

        # -----------------------
        # other variants
        try:
            thrown = False;
            midpoint(np.array([]), np.array([10, 10]));
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------

    def test_order_points(self):
        # -----------------------
        # classic test
        pts = np.array([[0, 5], [1, 5], [0, 4], [1, 4]])
        e = order_points(pts);
        npt.assert_array_equal(e, [[0, 5], [1, 5], [1, 4], [0, 4]]);
        # -----------------------
        # edge cases, that require an exception thrown
        thrown = False;
        # edge test: put in array (not ndarray)
        try:
            thrown = False;
            e = order_points([[0, 5], [1, 5], [0, 4], [1, 4]]);
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------
        # edge test: put in tuple (not ndarray)
        try:
            thrown = False;
            e = order_points(([0, 5], [1, 5], [0, 4], [1, 4]));
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------
        # edge test: put in ndarray, but less points than 4 (3 exactly)
        input = np.array([[0, 5], [1, 5], [0, 4]]);
        try:
            thrown = False;
            e = order_points(input);
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------
        # edge test: put in ndarray, but less points than 4 (2 exactly)
        input = np.array([[0, 5], [1, 5]]);
        try:
            thrown = False;
            e = order_points(input);
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------
        # edge test: put in ndarray, but less points than 4 (1 exactly)
        input = np.array([[0, 5]]);
        try:
            thrown = False;
            e = order_points(input);
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------
        # edge test: put in ndarray, but less points than 4 (0 exactly)
        input = np.array([]);
        try:
            thrown = False;
            e = order_points(input);
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------
        # edge test: put in ndarray, but all the points will be multi-dimensional
        input = np.array([[0, 0,0, 1, 2, 58], [0, 0, 0, 1, 2, 58], [0, 0,0, 1, 2, 58], [0, 0,0, 1, 2, 58]])
        try:
            thrown = False;
            e = order_points(input);
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------

    def test_artificial_circle_image(self):
        # -----------------------
        thrown = False;
        # edge test: put in an invalid size
        try:
            thrown = False;
            artificial_circle_image(-1);
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------
        # edge test: put in an invalid size
        try:
            thrown = False;
            artificial_circle_image(0.9);
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------
        # edge test: put in an invalid size
        try:
            thrown = False;
            artificial_circle_image(0);
        except ValueError:
            thrown = True;

        if not thrown:
            print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
            self.assertEqual(1, 0);
        # -----------------------

if __name__ == '__main__':
    unittest.main()
