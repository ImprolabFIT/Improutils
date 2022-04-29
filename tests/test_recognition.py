import unittest
from improutils.acquisition import *
from improutils.visualisation import *
from improutils.preprocessing import *
from improutils.segmentation import *
from improutils.recognition import *
from pathlib import Path

base_path = Path(__name__).parent.absolute() / 'tests' / 'img'

class ImageFeaturesTestCase(unittest.TestCase):
    # the following commented code can be used for executing the same test
    # upon all the functions, iterativelly. Traditionally, can be used for
    # edge case testing
    """
    def test_edge_cases_all(self):
        # load the test image, common for all the test cases
        img = load_image('{}/test-img.png'.format(base_path))
        img = to_hsv(img)
        # create a black image (segment nothing)
        img_bin = cv2.inRange(img, (0, 0, 0), (0, 0, 0))
        _, _, conts = find_contours(bin_im)
        # saving for debugging purposes
        # save_image(img_bin, '{}/out-img.png'.format(base_path))

        # create an array of tested functions, along with string name for
        # verbosity
        functions = [];
        functions.append((roundness, 'roundness'));
        functions.append((form_factor, 'form_factor'));
        functions.append((aspect_ratio, 'aspect_ratio'));
        functions.append((convexity, 'convexity'));
        functions.append((solidity, 'solidity'));
        functions.append((compactness, 'compactness'));
        functions.append((extent, 'extent'));
        # -----------------------
        # edge cases, that require exception throw

        for i in range(len(functions)):
            thrown = False;

            func, strName = functions[i];
            print('[TESTER]: testing {}'.format(strName));
            try:
                thrown = False;
                r = func(img_bin)
            except ValueError:
                thrown = True;

            if not thrown:
                print("An error was not thrown, expected the tested function to throw an error for th einput data provided.")
                self.assertEqual(1, 0);

        # -----------------------

    """
    def test_form_factor(self):
        eps = 0.04
        blue = ((90, 0, 0), (135, 255, 255))
        img = load_image('{}/test-img.png'.format(base_path))
        img = to_hsv(img)

        bin_im = cv2.inRange(img, blue[0], blue[1])
        _, _, conts = find_contours(bin_im)

        r = roundness(conts[0])
        self.assertLess(1.0 - r, eps)

    def test_aspect_ratio(self):
        eps = 0.04
        red = ((150, 0, 0), (180, 255, 255))
        img = load_image('{}/test-img.png'.format(base_path))
        img = to_hsv(img)

        img_bin = cv2.inRange(img, red[0], red[1])
        _, _, conts = find_contours(img_bin)
        ar = aspect_ratio(conts[0])

        self.assertLess(1.0 - ar, eps)

    def test_convexity(self):
        eps = 0.04
        red = ((150, 0, 0), (180, 255, 255))
        yellow = ((15, 0, 0), (30, 255, 255))
        img = load_image('{}/test-img.png'.format(base_path))
        img = to_hsv(img)

        img_bin_square = cv2.inRange(img, red[0], red[1])
        img_bin_star = cv2.inRange(img, yellow[0], yellow[1])
        _, _, conts_square = find_contours(img_bin_square)
        _, _, conts_star = find_contours(img_bin_star)

        c_square = convexity(conts_square[0])
        c_star = convexity(conts_star[0])
        self.assertLess(1.0 - c_square, eps)
        self.assertLess( c_star, eps*10)

    def test_solidity(self):
        eps = 0.041
        red = ((150, 0, 0), (180, 255, 255))
        yellow = ((15, 0, 0), (30, 255, 255))
        img = load_image('{}/test-img.png'.format(base_path))
        img = to_hsv(img)

        img_bin_square = cv2.inRange(img, red[0], red[1])
        img_bin_star = cv2.inRange(img, yellow[0], yellow[1])
        _, _, conts_square = find_contours(img_bin_square)
        _, _, conts_star = find_contours(img_bin_star)

        s_square = solidity(conts_square[0])
        s_star = solidity(conts_star[0])
        self.assertLess(1.0 - s_square, eps)
        self.assertLess(s_star, eps * 10)

    def test_compactness(self):
        eps = 0.04
        red = ((150, 0, 0), (180, 255, 255))
        yellow = ((15, 0, 0), (30, 255, 255))
        img = load_image('{}/test-img.png'.format(base_path))
        img = to_hsv(img)

        img_bin_square = cv2.inRange(img, red[0], red[1])
        img_bin_star = cv2.inRange(img, yellow[0], yellow[1])
        _, _, conts_square = find_contours(img_bin_square)
        _, _, conts_star = find_contours(img_bin_star)

        c_square = compactness(conts_square[0])
        c_star = compactness(conts_star[0])
        self.assertLess(1.0 - c_square, eps)
        self.assertLess( c_star, 0.6)

class OcrTestCase(unittest.TestCase):

    def test_ocr(self):
        img = load_image('{}/ocr-img.png'.format(base_path))
        img_bin = segmentation_two_thresholds(img, 0, 10)

        text = ocr(img_bin)
        # @FIXME
        # self.assertEqual(text, 'Improutils')

if __name__ == '__main__':
    unittest.main()
