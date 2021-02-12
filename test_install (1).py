def test_common_imports():
    """Test that basic libraries can be imported."""
    import numpy as np
    import cv2
    import matplotlib.pyplot as plt


def test_segmentation_one_threshold():
    """Test that segmentation_one_threshold can be imported and run."""
    import numpy as np
    from improutils import segmentation
    segmentation.segmentation_one_threshold(np.empty((200,200)), 125)


if __name__ == "__main__":
    test_common_imports()
    test_segmentation_one_threshold()
