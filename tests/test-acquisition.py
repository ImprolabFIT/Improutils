import unittest
from improutils.visualisation import *
import numpy as np

from improutils.acquisition import *

class ImgIOTestCase(unittest.TestCase):
    def test_load_image(self):
        img = load_image('../tests/img/test-img.png')
        self.assertEqual(img.shape, (746, 912, 3))

    def test_copy_to(self):
        src = load_image('../tests/img/test-img.png')
        dst = src.copy()
        dst[:500, :] = (0, 0, 0)
        mask = np.zeros((src.shape[0], src.shape[1]))
        mask[:500,:] = 1

        copy_to(src, dst, mask)

        self.assertFalse(np.bitwise_xor(src, dst ).any())

if __name__ == '__main__':
    unittest.main()
