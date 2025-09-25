import unittest
from pathlib import Path

import cv2
import numpy as np

from improutils import apply_fft, filtration_median, inverse_fft, load_image, to_gray

base_path = Path(__name__).parent.absolute() / "tests" / "img"


class FiltrationTestCase(unittest.TestCase):
    # https://github.com/opencv/opencv/blob/master/modules/python/test/test_dft.py

    def test_inverse_fft(self):
        eps = 0.01
        img = load_image(str(base_path / "test-img.png"))
        img = to_gray(img)
        ref_magnitude, fft_shift = apply_fft(img)
        img_inverse = inverse_fft(fft_shift)

        test_dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        img_back = cv2.idft(test_dft)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        img_back = cv2.normalize(img_back, 0.0, 1.0, cv2.NORM_MINMAX)
        img_inverse = cv2.normalize(img_inverse, 0.0, 1.0, cv2.NORM_MINMAX)

        self.assertLess(cv2.norm(img_inverse - img_back), eps)

    def test_filtration_median(self):
        img = load_image(str(base_path / "test-img.png"))
        img_median = filtration_median(img, 5)

        self.assertEqual(img.shape, img_median.shape)


if __name__ == "__main__":
    unittest.main()
