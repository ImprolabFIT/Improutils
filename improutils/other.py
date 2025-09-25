import os

import cv2
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display


def midpoint(ptA, ptB):
    """Return the midpoint between two points.

    Parameters
    ----------
    ptA : array | tuple | ndarray
        The first 2D point considered
    ptB : array | tuple | ndarray
        The second 2D point considered

    Returns
    -------
    The 2D midpoint

    """
    return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def artificial_circle_image(size):
    """Create an image of given size filled with circles.

    Parameters
    ----------
    size : int
        size of the image

    Returns
    -------
    Artificial image with circles

    """
    img_art_circ = np.zeros((int(size), int(size)), dtype=np.uint8)
    step = 10
    for i in range(step, int(size), step):
        cv2.circle(
            img_art_circ,
            (int(size / 2.0), int(size / 2.0)),
            i - step,
            np.random.randint(0, 255),
            thickness=4,
        )
    return img_art_circ


def order_points(pts):
    """Sort the points based on their coordinates, in top-left, top-right, bottom-right, and bottom-left order.

    Parameters
    ----------
    pts : ndarray
        4 2D Points to be sorted.

    Returns
    -------
        Sorted points, the coordinates in top-left, top-right, bottom-right, and bottom-left order

    """
    if not isinstance(pts, np.ndarray):
        raise ValueError(
            "Invalid input point format. Numpy ndarray expected. Got {}".format(
                type(pts)
            )
        )

    if len(pts) < 4:
        raise ValueError(
            "Invalid amount of input points. Got {} elements".format(len(pts))
        )

    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (bl, tl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    rightMost = rightMost[np.argsort(rightMost[:, 1]), :]
    (br, tr) = rightMost

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")


def pcd_to_depth(pcd, height, width):
    """Reduce point-cloud to coordinates, point cloud [x, y, z, rgb] -> depth[x, y, z].

    Parameters
    ----------
    pcd : array
        point cloud
    height : int
        height of captured img
    width : int
        width of a captured img

    Returns
    -------
    Array of coordinates.

    """
    data = pcd
    data = [float(x.split(" ")[2]) for x in data]
    data = np.reshape(data, (height, width))
    return data


def create_file_path(folder, file_name):
    """Create path for filename inside a folder.

    Parameters
    ----------
    folder : string
        Base folder directory in string notation.
        If the directory does not exist, it is created.

    file_name : string
        File name that should be inside the base folder.

    Returns
    -------
    Path to the file.

    """
    if not os.path.isdir(folder):
        os.mkdir(folder)

    return os.path.join(folder, file_name)


def create_slider(min, max, description):
    """Create an integer range slider widget.

    Parameters
    ----------
    min : int
        The minimum value for the range slider.
    max : int
        The maximum value for the range slider.
    description : str
        The label to display alongside the slider.

    Returns
    -------
    ipywidgets.IntRangeSlider
        An IntRangeSlider widget with the specified bounds and label.

    """
    description = description.ljust(30, "\xa0")
    return widgets.IntRangeSlider(
        min=min,
        max=max,
        step=1,
        value=[min, max],
        description=description,
        continuous_update=False,
        orientation="horizontal",
        style=dict(description_width="initial"),
        layout=widgets.Layout(width="auto"),
    )


def multicolor_segmentation(func, colors):
    """Interactive HSV thresholding for multiple colors with saving and returning thresholds that are picked by the user.

    Parameters
    ----------
    func : function
        function with arguments hue = h_range (int, range: 0-360), saturation = s_range (int, range: 0-255), value = v_range (int, range: 0-255)
    colors : list
        list of colors that the user can choose from, e.g. ['red', 'green', 'blue'], these colors will be used as keys in the output dictionary

    Returns
    -------
    color_thresholds: dict
        Returns a dictionary with the chosen thresholds for each color, e.g. {'red': (0, 0, 0), 'green': (0, 0, 0), 'blue': (0, 0, 0)}, can be also empty if no thresholds were saved

    """
    color_thresholds = {}

    # initialize sliders, buttons etc.
    h_slider = create_slider(min=0, max=360, description="Hue:")
    s_slider = create_slider(min=0, max=255, description="Saturation:")
    v_slider = create_slider(min=0, max=255, description="Value:")

    color_dropdown = widgets.Dropdown(
        options=colors,
        description="Color:".ljust(30, "\xa0"),
        style={"description_width": "initial"},
        layout={"width": "max-content"},
    )

    save_button = widgets.Button(
        description="Save threshold for color",
        layout=widgets.Layout(width="auto"),
        button_style="success",
    )
    finish_button = widgets.Button(
        description="Return saved thresholds",
        layout=widgets.Layout(width="auto"),
        button_style="danger",
    )

    text_output = widgets.Output()
    interactive_output = widgets.interactive_output(
        func, {"h_range": h_slider, "s_range": s_slider, "v_range": v_slider}
    )

    # widget layout
    input_box = widgets.VBox([h_slider, s_slider, v_slider, color_dropdown])
    button_box = widgets.HBox([save_button, finish_button])
    other_box = widgets.VBox([text_output, interactive_output])

    def reset_sliders():
        h_slider.value = (0, 360)
        s_slider.value = (0, 255)
        v_slider.value = (0, 255)

    # button callbacks
    def on_save_clicked(b):
        with text_output:
            text_output.clear_output()
            color_thresholds[color_dropdown.value] = (
                h_slider.value,
                s_slider.value,
                v_slider.value,
            )
            print(
                f"Saved for color '{color_dropdown.value}', threshold: {color_thresholds[color_dropdown.value]}\nResetting sliders...\nChanging to next color..."
            )
            reset_sliders()
            # set next color in dropdown
            color_dropdown.value = colors[
                (colors.index(color_dropdown.value) + 1) % len(colors)
            ]

    def on_finish_clicked(b):
        with text_output:
            text_output.clear_output()
            print("Returned saved thresholds!")
            reset_sliders()

    save_button.on_click(on_save_clicked)
    finish_button.on_click(on_finish_clicked)
    # display widget
    display(input_box, button_box, other_box)

    return color_thresholds


def _plot_grid(xv, yv, squares, ax):
    for i in np.linspace(0, xv.shape[1] - 1, squares + 1, dtype=int):
        ax.plot(xv[i, :], yv[i, :], "k-")
    for j in np.linspace(0, xv.shape[0] - 1, squares + 1, dtype=int):
        ax.plot(xv[:, j], yv[:, j], "k-")

    ax.axis("off")


def _radial_distortion(xv, yv, k):
    xv_radial = np.zeros_like(xv)
    yv_radial = np.zeros_like(yv)
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            r = np.sqrt(xv[i, j] ** 2 + yv[i, j] ** 2)
            radial = (1 + (k[0] * (r**2) + k[1] * (r**4) + k[2] * (r**6))) / (
                1 + (k[3] * (r**2) + k[4] * (r**4) + k[5] * (r**6))
            )
            xv_radial[i, j] = xv[i, j] * radial
            yv_radial[i, j] = yv[i, j] * radial
    return xv_radial, yv_radial


def _tangetial_distortion(xv, yv, p):
    xv_tang = np.zeros_like(xv)
    yv_tang = np.zeros_like(yv)
    for i in range(xv.shape[0]):
        for j in range(xv.shape[1]):
            x = xv[i, j]
            y = yv[i, j]
            r = np.sqrt(x**2 + y**2)
            x_tang = x + (2 * p[0] * x * y + p[1] * (r**2 + 2 * x**2))
            y_tang = y + (p[0] * (r**2 + 2 * y**2) + 2 * p[1] * x * y)
            xv_tang[i, j] = x_tang
            yv_tang[i, j] = y_tang
    return xv_tang, yv_tang


def plot_distortion(
    k1: float,
    k2: float,
    k3: float,
    k4: float,
    k5: float,
    k6: float,
    p1: float,
    p2: float,
) -> None:
    """Vizualization of distortion parameters and their influence. Plots radial, tangential and compounded (radial + tangential) distortion grid using the Brown-Conrady (OpenCV) model.

    Parameters
    ----------
    k1 : float
        Radial distortion coefficient.
    k2 : float
        Radial distortion coefficient.
    k3 : float
        Radial distortion coefficient.
    k4 : float
        Radial distortion coefficient.
    k5 : float
        Radial distortion coefficient.
    k6 : float
        Radial distortion coefficient.
    p1 : float
        Tangential distortion coefficient.
    p2 : float
        Tangential distortion coefficient.

    Returns
    -------
    None

    """
    k = (k1, k2, k3, k4, k5, k6)
    p = (p1, p2)
    squares = 10  # amount of squares in the grid
    pts = 100
    # realistical values for image with 2500 x 2500 pixels with focal length of 35mm which is close to 10500 pixels with basler camera pixel size, origin is in the center - therfore x and y should be within values +-(2500/10500)/2
    width = 0.23
    height = 0.23
    xv, yv = np.meshgrid(
        np.linspace(-width / 2, width / 2, pts),
        np.linspace(-height / 2, height / 2, pts),
    )

    xv_radial, yv_radial = _radial_distortion(xv, yv, k)
    xv_tang, yv_tang = _tangetial_distortion(xv, yv, p)

    _, axs = plt.subplots(1, 3, figsize=(15, 5))

    _plot_grid(xv_radial, yv_radial, squares, axs[0])
    axs[0].set_title("Radial distortion grid")

    _plot_grid(xv_tang, yv_tang, squares, axs[1])
    axs[1].set_title("Tangential distortion grid")

    _plot_grid(xv_radial + xv_tang, yv_radial + yv_tang, squares, axs[2])
    axs[2].set_title("Compounded distortion grid")
    plt.show()
