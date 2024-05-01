import PIL
from pytesseract import pytesseract

from improutils import negative


def ocr(img_bin, config='', lang=None):
    """
    Detects text in the file.

    Parameters
    ----------
    img_bin : ndarray
        Input binary image. White objects on black background.
    config : str
        Model config, refer to: https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html,
        https://muthu.co/all-tesseract-ocr-options/ for correct use.
        Defaults to ''.
    lang : str | None
        Language code, e.g. `eng` for English and `ces` for Czech. For list of language codes, refer to:
        https://tesseract-ocr.github.io/tessdoc/Data-Files-in-different-versions.html.
        Selected language must be installed using `sudo apt-get install tesseract-ocr-langcode`
        where `langcode` is the language code. English is installed by default.
        Defaults to None.
    Returns
    -------
    The recognized text in the image.
    """
    # Tesseract works with black objects on white background.
    img_bin = negative(img_bin)
    return pytesseract.image_to_string(PIL.Image.fromarray(img_bin), config=config, lang=lang)
