# image feature extraction functions and classes
from .image_features import (
    ShapeDescriptors,
    aspect_ratio,
    compactness,
    convexity,
    extent,
    form_factor,
    roundness,
    solidity,
)

# OCR (Optical Character Recognition) functions
from .ocr import (
    ocr,
)

# QR code detection and decoding functions
from .qr import (
    qr_decode,
    qr_detect,
    qr_detect_and_decode,
    qr_init_reader,
)

# define public API
__all__ = [
    # image feature extraction
    "ShapeDescriptors",
    "form_factor",
    "roundness",
    "aspect_ratio",
    "convexity",
    "solidity",
    "compactness",
    "extent",
    # OCR functions
    "ocr",
    # QR code functions
    "qr_init_reader",
    "qr_detect_and_decode",
    "qr_decode",
    "qr_detect",
]
