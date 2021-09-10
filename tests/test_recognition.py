import unittest
from improutils.acquisition import *
from improutils.visualisation import *
from improutils.preprocessing import *
from improutils.segmentation import *
from improutils.recognition import *
from pathlib import Path

base_path = Path(__name__).parent.absolute() / 'tests' / 'img'


class ImageFeaturesTestCase(unittest.TestCase):

    def test_form_factor(self):
        eps = 0.04
        blue = ((90, 0, 0), (135, 255, 255))
        img = load_image(str(base_path / 'test-img.png'))
        img = to_hsv(img)

        img_bin = cv2.inRange(img, blue[0], blue[1])

        r = roundness(img_bin)
        self.assertLess(1.0 - r, eps)

    def test_aspect_ratio(self):
        eps = 0.04
        red = ((150, 0, 0), (180, 255, 255))
        img = load_image(str(base_path / 'test-img.png'))
        img = to_hsv(img)

        img_bin = cv2.inRange(img, red[0], red[1])
        ar = aspect_ratio(img_bin)

        self.assertLess(1.0 - ar, eps)

    def test_convexity(self):
        eps = 0.04
        red = ((150, 0, 0), (180, 255, 255))
        yellow = ((15, 0, 0), (30, 255, 255))
        img = load_image(str(base_path / 'test-img.png'))
        img = to_hsv(img)

        img_bin_square = cv2.inRange(img, red[0], red[1])
        img_bin_star = cv2.inRange(img, yellow[0], yellow[1])

        c_square = convexity(img_bin_square)
        c_star = convexity(img_bin_star)
        self.assertLess(1.0 - c_square, eps)
        self.assertLess( c_star, eps*10)

    def test_solidity(self):
        eps = 0.041
        red = ((150, 0, 0), (180, 255, 255))
        yellow = ((15, 0, 0), (30, 255, 255))
        img = load_image(str(base_path / 'test-img.png'))
        img = to_hsv(img)

        img_bin_square = cv2.inRange(img, red[0], red[1])
        img_bin_star = cv2.inRange(img, yellow[0], yellow[1])

        s_square = solidity(img_bin_square)
        s_star = solidity(img_bin_star)
        self.assertLess(1.0 - s_square, eps)
        self.assertLess(s_star, eps * 10)

    def test_compactness(self):
        eps = 0.04
        red = ((150, 0, 0), (180, 255, 255))
        yellow = ((15, 0, 0), (30, 255, 255))
        img = load_image(str(base_path / 'test-img.png'))
        img = to_hsv(img)

        img_bin_square = cv2.inRange(img, red[0], red[1])
        img_bin_star = cv2.inRange(img, yellow[0], yellow[1])

        c_square = compactness(img_bin_square)
        c_star = compactness(img_bin_star)
        self.assertLess(1.0 - c_square, eps)
        self.assertLess( c_star, 0.6)

class OcrTestCase(unittest.TestCase):

    def test_ocr(self):
        img = load_image(str(base_path / 'ocr-img.png'))
        img_bin = segmentation_two_thresholds(img, 0, 10)

        text = ocr(img_bin)
        self.assertEqual(text, 'Improutils')

if __name__ == '__main__':
    unittest.main()
