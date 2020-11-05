import PIL
from pytesseract import pytesseract

from improutils import negative


def ocr(img_bin):
    '''Detects text in the file.

    Parameters
    ----------
    img_bin : numpy.ndarray
        Input binary image. White objects on black background.
    Returns
    -------
    Text on image.
    '''
    # Tesseract works with black objects on white background.
    img_bin = negative(img_bin)
    return pytesseract.image_to_string(PIL.Image.fromarray(img_bin))