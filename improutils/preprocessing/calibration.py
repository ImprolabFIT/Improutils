import os
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import yaml
from prettytable import PrettyTable

# consts
IDX_CAM_MATRIX = "camera_matrix"
IDX_DIST_COEFFS = "dist_coefs"


def camera_calibration(
    calib_path: str,
    chess_shape: Tuple[int, int],
    cv2_flags: int = 0,
    extensions: List[str] = ["jpg", "jpeg", "png", "tiff", "bmp"],
) -> Tuple[
    float,
    np.ndarray,
    np.ndarray,
    List[np.ndarray],
    List[np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Dict[str, np.ndarray],
]:
    """Calibrate camera from images with chessboard pattern, using OpenCV's cv2.calibrateCameraExtended function.

    Parameters
    ----------
    calib_path : str
        Path to the folder containing chessboard pattern images.
    chess_shape : Tuple[int, int]
        Interior corner count in the format of rows, columns.
    cv2_flags : int, optional
        Additional OpenCV flags for cv2.calibrateCameraExtended. Defaults to 0.
    extensions : List[str], optional
        Allowed image extensions. Defaults to ["jpg", "jpeg", "png", "tiff"].

    Returns
    -------
    Tuple[float, np.ndarray, np.ndarray, Tuple[np.ndarray], Tuple[np.ndarray], np.ndarray, np.ndarray, np.ndarray, Dict[str,np.ndarray]]
        Returns the output from cv2.calibrateCameraExtended and a dictionary with image names as keys and images with drawn chessboard corners as values.

    Raises
    ------
    ValueError
        If calibration images have different sizes.
    ValueError
        If no calibration images were found or could not be read from the provided path.
    ValueError
        If no chessboard patterns were detected in the images.

    """
    print(f"Processing images from {calib_path} with possible extensions {extensions}")

    def correct_extension(path, extensions):
        return path.is_file() and path.suffix[1:].lower() in extensions

    # termination criteria for subpixel corner detection
    # by default it is set to 30 iterations and epsilon = 0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((chess_shape[0] * chess_shape[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : chess_shape[0], 0 : chess_shape[1]].T.reshape(-1, 2)

    # arrays to store object points and image points from all the images.
    objpoints = []  # 3D point in real world space
    imgpoints = []  # 2D points in image plane.

    image_paths = sorted(
        [
            path
            for path in Path(calib_path).glob("*")
            if correct_extension(path, extensions)
        ]
    )
    chess_brd_images = 0
    read_images = 0
    chessboard_images = {}
    img_size = None
    for img_path in image_paths:
        img_name = img_path.name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"File {img_name} could not be read, skipping...")
            continue
        else:
            read_images += 1
            if img_size is None:
                # need to be in the format of width, height
                img_size = img.shape[:2][::-1]
            else:
                if img_size != img.shape[:2][::-1]:
                    raise ValueError("All images must have same size.")
            print(f"File {img_name} is being processed...")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, chess_shape, None)

        # if found, add object points, image points (after refining them)
        if ret:
            chess_brd_images += 1
            print("\t Corners found!")
            objpoints.append(objp)
            subpix_corners = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), criteria
            )
            imgpoints.append(subpix_corners)

            chessboard_images[img_name] = cv2.drawChessboardCorners(
                img, chess_shape, subpix_corners, ret
            )

        else:
            print("\t Corners NOT found!")
            continue

    print(
        f"Number of images with detected chessboard: {chess_brd_images}/{read_images}"
    )

    if read_images == 0:
        raise ValueError("No images were read from the provided path.")

    if chess_brd_images == 0:
        raise ValueError("No chessboard patterns were detected in the images.")

    calib_values = cv2.calibrateCameraExtended(
        objpoints,
        imgpoints,
        img_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=cv2_flags,
        criteria=criteria,
    )
    # calib values is a tuple of:
    # reprojection_error
    # camera_matrix
    # dist_coeffs
    # rvecs
    # tvecs
    # std_deviations_intrinsics
    # std_deviations_extrinsics
    # per_view_errors
    
    reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs, std_deviations_intrinsics, std_deviations_extrinsics, per_view_errors = calib_values
    return reprojection_error, camera_matrix, dist_coeffs, rvecs, tvecs, std_deviations_intrinsics, std_deviations_extrinsics, per_view_errors, chessboard_images
    # return calib_values  # type:ignore


def calibration_stats(
    reprojection_error: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    std_deviations_intrinsics: np.ndarray | None = None,
    per_view_errors: np.ndarray | None = None,
    view_names: List[str] | None = None,
    pixel_size: Union[float, Tuple[float, float]] | None = None,
) -> None:
    """Print calibration statistics.

    RMS re-projection error, estimated intrinsics and distortion parameters, with standard deviations,
    focal length in millimeters, and per-view reprojection errors.

    Parameters
    ----------
    reprojection_error : float
        Re-projection error from cv2.calibrateCamera.
    camera_matrix : np.ndarray
        Camera matrix from cv2.calibrate.
    dist_coeffs : np.ndarray
        Distortion coefficients from cv2.calibrateCamera.
    std_deviations_intrinsics : np.ndarray, optional
        Standard deviations of intrinsics from cv2.calibrateCameraExtended. Defaults to None.
    per_view_errors : np.ndarray, optional
        Per-view errors from cv2.calibrateCameraExtended. Defaults to None.
    view_names : List[str], optional
        Image names for which the chessboard was detected. Defaults to None.
    pixel_size : Union[float, Tuple[float, float]], optional
        Size of physical pixels of a camera in micrometers (e.g., 4.8, 5.86, or [5.86, 4.8] for non-square pixels). Defaults to None.

    """
    # opencv always returns atleast 4 distortion coefficients
    params_amount = 4 + dist_coeffs.shape[1]

    parameters = [
        "fx",
        "fy",
        "cx",
        "cy",
        "k1",
        "k2",
        "p1",
        "p2",
        "k3",
        "k4",
        "k5",
        "k6",
        "s1",
        "s2",
        "s3",
        "s4",
        "Tx",
        "Ty",
    ]
    units = ["pixels"] * 4 + ["unitless"] * (params_amount - 4)

    print(f"RMS re-projection error: {reprojection_error:.5f} pixels")

    print("\nEstimated intrinsics parameters")
    if std_deviations_intrinsics is None:
        print("\tNo standard deviations provided!")
        std_deviations_intrinsics = np.zeros(params_amount)

    intrinsics_table = PrettyTable()
    intrinsics_table.add_column("Parameter", parameters[:4])
    intrinsics_table.add_column(
        "Estimated Value ± Std Deviation",
        [
            f"{val:.5f} ± {std:.5f}"
            for val, std in zip(
                [
                    camera_matrix[0, 0],
                    camera_matrix[1, 1],
                    camera_matrix[0, 2],
                    camera_matrix[1, 2],
                ],
                std_deviations_intrinsics[:4, 0],
            )
        ],
    )
    intrinsics_table.add_column("Unit", units[:4])
    print(intrinsics_table)

    print("\nEstimated Distortion parameters")
    distortion_table = PrettyTable()
    distortion_table.add_column("Parameter", parameters[4:params_amount])
    distortion_table.add_column(
        "Distortion ± Std Deviation",
        [
            f"{val:.5f} ± {std:.5f}"
            for val, std in zip(
                dist_coeffs[0, : params_amount - 4],
                std_deviations_intrinsics[4 : params_amount + 4, 0],
            )
        ],
    )
    distortion_table.add_column("Unit", units[4:params_amount])
    print(distortion_table)

    if pixel_size is not None and std_deviations_intrinsics is not None:
        if not isinstance(pixel_size, tuple):
            pixel_size = (pixel_size, pixel_size)
        print("\nEstimated Focal length in millimeters")
        focal_length_table = PrettyTable()
        focal_length_table.add_column("Parameter", parameters[:2])
        focal_length_table.add_column(
            "Value ± Std Deviation",
            [
                f"{val * pix_size / 1000:.5f} ± {std * pix_size / 1000:.5f}"
                for val, pix_size, std in zip(
                    [camera_matrix[0, 0], camera_matrix[1, 1]],
                    pixel_size,
                    std_deviations_intrinsics[:2, 0],
                )
            ],
        )
        focal_length_table.add_column("Unit", ["millimeter"] * 2)
        print(focal_length_table)

    if per_view_errors is not None and view_names is not None:
        print("\nPer view reprojection errors")
        view_error_table = PrettyTable()
        # Sort the view names and errors by the errors in descending order
        sorted_views_and_errors = sorted(
            zip(view_names, per_view_errors[:, 0]), key=lambda x: x[1], reverse=True
        )
        sorted_view_names, sorted_errors = zip(*sorted_views_and_errors)
        view_error_table.add_column("Image name", sorted_view_names)
        view_error_table.add_column(
            "Re-projection error (sorted)", [f"{val:.5f}" for val in sorted_errors]
        )
        view_error_table.add_column("Unit", ["pixels"] * len(sorted_view_names))
        print(view_error_table)


def correct_frame(frame, camera_matrix, dist_coeffs):
    """Return an undistorted frame."""
    return cv2.undistort(frame, camera_matrix, dist_coeffs)


def load_camera_calib(input_file):
    """Load camera calibration from specified input file.

    Parameters
    ----------
    input_file : string
        Input file with calibration data in YAML format.

    Returns
    -------
    tuple(ndarray, ndarray)
        Returns a tuple where first element is camera matrix array and second element is dist coefficients array.
        These arrays might be empty if the file isn't found or in correct format.

    """
    try:
        with open(input_file, "r") as stream:
            data = yaml.safe_load(stream)
            return data[IDX_CAM_MATRIX], data[IDX_DIST_COEFFS]
    except (FileNotFoundError, yaml.YAMLError):
        print(f"File {input_file} couldn't be read.")
        return np.array([]), np.array([])


def save_camera_calib(output_file, camera_matrix, dist_coefs):
    """Save camera calibration data to a YAML file.

    This function stores the camera matrix and distortion coefficients in
    the specified output file. If the parent directory does not exist, it
    is created automatically.

    Parameters
    ----------
    output_file : str
        Path to the output YAML file where calibration data will be saved.
    camera_matrix : ndarray
        Camera intrinsic matrix.
    dist_coefs : ndarray
        Distortion coefficients of the camera.

    Returns
    -------
    None

    """
    data = {IDX_CAM_MATRIX: camera_matrix, IDX_DIST_COEFFS: dist_coefs}
    output_dir = os.path.dirname(output_file)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    with open(output_file, "w") as f:
        yaml.dump(data, f)
