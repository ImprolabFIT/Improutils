import copy
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import NoNorm, Normalize

from improutils.acquisition import copy_to
from improutils.other import midpoint, order_points
from improutils.segmentation import to_3_channels


def plot_images(
    *imgs, titles=[], channels="bgr", normalize=False, ticks_off=True, title_size=32
):
    """Plot multiple images in one figure.

    Parameters
    ----------
    *imgs : list
        Arbitrary number of  images to be shown.
    titles : list
        Titles for each image.
    channels : string
        Colour channels. Possible values are "bgr", "rgb" or "mono".
    normalize : bool
        If True, image will be normalized.
    ticks_off : bool
        If True, axis decorations will be hidden.
    title_size : int
        Size of the title.

    Returns
    -------
    None

    """
    assert channels.lower() in ["bgr", "rgb", "mono"], (
        "Possible values for channels are: bgr, rgb or mono!"
    )

    #     f = plt.figure(figsize=(30, 20))
    width_def = 60
    height_def = 60

    width = math.ceil(math.sqrt(len(imgs)))
    height = math.ceil(len(imgs) / width)

    height_def = height_def / 5 * width
    #     print(height_def)
    if height_def > 65:
        height_def = 65

    f = plt.figure(figsize=(width_def, height_def))

    #     print(str(width) + ' , ' + str(height))
    for i, img in enumerate(imgs, 1):
        ax = f.add_subplot(height, width, i)
        if ticks_off:
            ax.axis("off")

        if len(titles) != 0:
            if len(imgs) != len(titles):
                print("WARNING titles length is not the same as images length!")

            try:
                ax.set_title(
                    str(titles[i - 1]),
                    fontdict={"fontsize": title_size, "fontweight": "medium"},
                )
            except IndexError:
                pass

        if channels.lower() == "mono" or img.ndim == 2:
            if normalize:
                norm = Normalize()
            else:
                norm = NoNorm()
            ax.imshow(img, cmap=plt.get_cmap("gray"), norm=norm)
        elif channels.lower() == "rgb":
            ax.imshow(img)
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def show_images(*imgs, scale=1, window_name="Image preview"):
    """Display multiple images in separate resizable windows.

    Each image in the input list is shown in its own window. The user can
    click on the images to print the (x, y) coordinates of mouse clicks.
    The preview is terminated by pressing the 'q', 'Q', or 'Esc' key.

    Parameters
    ----------
    *imgs : ndarray
        One or more images to display. Each image can be grayscale or color (BGR).
    scale : float, optional
        Scaling factor for the displayed image windows. Default is 1.
    window_name : str, optional
        Base name for the displayed windows. Default is "Image preview".

    Returns
    -------
    None

    Notes
    -----
    Known bug for Mac users: see
    https://gitlab.fit.cvut.cz/bi-svz/bi-svz/issues/13

    """

    def print_xy(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print("x = %d, y = %d" % (x, y))

    for i, img in enumerate(imgs, 1):
        h, w = img.shape[:2]
        window_name_id = window_name + " " + str(i)
        cv2.namedWindow(window_name_id, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name_id, int(w * scale), int(h * scale))
        cv2.setMouseCallback(window_name_id, print_xy)
        cv2.moveWindow(window_name_id, (i - 1) * int(w * scale), 0)

    while 1:
        for i, img in enumerate(imgs, 1):
            cv2.imshow(window_name + " " + str(i), img)

        k = cv2.waitKey(0)

        if k == ord("q") or k == ord("Q") or k == 27:
            break

    cv2.destroyAllWindows()


def show_camera_window(*imgs, scale=1):
    """Open input images in separate windows.

    Parameters
    ----------
    *imgs : list
        Arbitrary number of images to be shown.
    scale : double
        Scale of shown image window.

    Returns
    -------
    None

    """

    def print_xy(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            print("x = %d, y = %d" % (x, y))

    for i, img in enumerate(imgs, 1):
        window_name_id = "Camera capture" + " " + str(i)

        h, w = img.shape[:2]
        cv2.namedWindow(window_name_id, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name_id, int(w * scale), int(h * scale))
        cv2.setMouseCallback(window_name_id, print_xy)
        if len(imgs) > 1:
            cv2.moveWindow(window_name_id, (i - 1) * int(w * scale), 0)
        cv2.imshow(window_name_id, img)


def draw_rotated_rect(img, cnt):
    """Draw rotated rectangle with minimum area into the image, around the contour.

    Input image is not modified.

    Parameters
    ----------
    img : ndarray
        Input image.
    cnt : ndarray
        Contour around which the rectangle will be drawn

    Returns
    -------
    res : ndarray
        Image with drawn rectangle on it.
    rect : ndarray
        rectangle from cv2.minAreaRect

    """
    res = img.copy()
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.array(box).astype(np.int32)
    cv2.drawContours(res, [box], -1, (255, 255, 255), 1)
    return res, rect


def draw_text(img, text, point, text_scale, text_color, text_thickness):
    """Draw rotated text into the image.

    Parameters
    ----------
    img : ndarray
        Input image.
    text : string
        Text to be drawn.
    point : tuple
        Point where text is drawn.
    text_scale : double
        Scale of text.
    text_color : tuple
        Color of text.
    text_thickness : int
        Thickness of text.

    Returns
    -------
    Output image.

    """
    img_text = copy.deepcopy(img)
    # create rotated text mask
    cv2.putText(
        img_text,
        "{:.2f} cm".format(text),
        point,
        0,
        text_scale,
        text_color,
        text_thickness,
    )
    return img_text


def draw_real_sizes(
    img,
    rect,
    width_text,
    height_text,
    lbl_size_scale=2,
    lbl_color=(0, 0, 255),
    lbl_thickness=8,
):
    """Draw real sizes of rotated rectangle into the image.

    Parameters
    ----------
    img : ndarray
        Input image.
    rect : tuple
        Rotated rectangle.
    width_text : string
        Width of the rectangle in the form of string.
    height_text : string
        Height of the rectangle in the form of string.
    lbl_size_scale : double
        Scale of text.
    lbl_color : tuple
        Color of text.
    lbl_thickness : int
        Thickness of text.

    Returns
    -------
    Output image.

    """
    tl, tr, br, bl = order_points(cv2.boxPoints(rect))
    mid_pt_height = midpoint(tl, bl)
    mid_pt_width = midpoint(bl, br)

    # bottom-left points where labels are drawn
    pt_label_first = (int(mid_pt_width[0] - 10), int(mid_pt_width[1] - 10))
    pt_label_second = (int(mid_pt_height[0] + 10), int(mid_pt_height[1]))

    result = draw_text(
        img,
        width_text,
        pt_label_first,
        lbl_size_scale,
        lbl_color,
        lbl_thickness,
    )
    result = draw_text(
        result,
        height_text,
        pt_label_second,
        lbl_size_scale,
        lbl_color,
        lbl_thickness,
    )
    return result


def draw_lines(img, lines):
    """Draw lines coming from  HoughLines procedure into image.

    Parameters
    ----------
    img : ndarray
        Input image.
    lines : ndarray
        array of lines - output of cv2.HoughLines.

    Returns
    -------
    Output image.

    """
    img_lines = to_3_channels(img)

    for line in lines:
        line = line[0]
        cv2.line(
            img_lines,
            (line[0], line[1]),
            (line[2], line[3]),
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return img_lines


def color_picker(img):
    """Interactive tool to pick colors from an image.

    Displays the input image in a window and allows the user to click
    on points to sample their colors. Each selected color is printed
    in the console and annotated on the image. Press 'q', 'Q', or
    'Esc' to exit.

    Parameters
    ----------
    img : ndarray
        Input image in which colors will be sampled. Can be grayscale
        or color (BGR) image.

    Returns
    -------
    None
        This function does not return a value. Selected colors are printed
        to the console and annotated on the displayed image.

    """
    img = img.copy()
    window_name = "color picker"
    colors = []

    def on_mouse_click(event, x, y, _, img):
        if event == cv2.EVENT_LBUTTONUP:
            colors.append(img[y, x].tolist())
            cv2.putText(
                img,
                f"Point {len(colors)}: {colors[-1]}",
                (10, 50),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (0, 0, 0),
                2,
            )
            print(f"Point {len(colors)}: {colors[-1]}")

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse_click, img)

    while True:
        cv2.imshow(window_name, img)
        k = cv2.waitKey(0)
        if k == ord("q") or k == ord("Q") or k == 27:
            break

    cv2.destroyAllWindows()
