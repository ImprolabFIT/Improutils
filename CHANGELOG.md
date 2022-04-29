 - recognition, shape descriptor functions (`\recognition\image_features.py`) no longer crash, when no contour is found
 in the source image. Instead, a ValueError exception is thrown.
 This applies to following functions:
    - roundness
    - form_factor
    - aspect_ratio
    - convexity
    - solidity
    - compactness
    - extent
 - add tests for the behavior described above
 - add more comments to functions in `\recognition\image_features.py`
 - correct grammar in descriptions and text
 - make the string concatenation more readable
