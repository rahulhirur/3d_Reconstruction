import open3d as o3d
from open3d import geometry as o3dg
import json
import numpy as np
import cv2

import datetime
import sys

import plotly.graph_objects as go
import plotly.express as px

from scipy.ndimage import zoom
import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_camera_intrinsics(calibration_data, camera_index):
    """
    Extracts and formats camera calibration data for a specific camera index.

    Args:
        calibration_data: The loaded calibration data (JSON dictionary).
        camera_index: The index of the camera (0 for the first, 1 for the second, etc.).

    Returns:
        K: Intrinsic matrix (3x3 numpy array).
        D: Distortion coefficients (numpy array).
        image_size: Tuple (width, height).
    """

    camera_data = calibration_data["calibration"]["cameras"][camera_index]
    parameters = camera_data["model"]["ptr_wrapper"]["data"]["parameters"]

    # Extract intrinsic parameters
    fx = parameters["f"]["val"]
    fy = fx / parameters["ar"]["val"]
    cx = parameters["cx"]["val"]
    cy = parameters["cy"]["val"]

    # Extract distortion coefficients
    k1 = parameters["k1"]["val"]
    k2 = parameters["k2"]["val"]
    k3 = parameters["k3"]["val"]
    p1 = parameters["p1"]["val"]
    p2 = parameters["p2"]["val"]

    # Create intrinsic matrix
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Create distortion coefficient array
    D = np.array([k1, k2, p1, p2, k3])

    return K, D

def load_camera_extrinsics(calibration_data,camera_index=1):
    """
    Extracts stereo calibration data.

    Args:
        calibration_data: The loaded calibration data (JSON dictionary).
        second_camera_index: The index of the second camera, from which the rotation and translation are extracted.

    Returns:
        R: Rotation matrix (3x3 numpy array).
        T: Translation vector (3x1 numpy array).
    """

    transform_data = calibration_data["calibration"]["cameras"][camera_index]["transform"]
    rx = transform_data["rotation"]["rx"]
    ry = transform_data["rotation"]["ry"]
    rz = transform_data["rotation"]["rz"]
    R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))

    tx = transform_data["translation"]["x"]
    ty = transform_data["translation"]["y"]
    tz = transform_data["translation"]["z"]
    T = np.array([[tx], [ty], [tz]])

    return R, T

def load_and_print_calibration(json_file_path):
    with open(json_file_path, 'r') as f:
        calibration_data = json.load(f)

    K1, D1 = load_camera_intrinsics(calibration_data, 0)
    K2, D2 = load_camera_intrinsics(calibration_data, 1)
    R, T = load_camera_extrinsics(calibration_data, 1)

    return K1, D1, K2, D2, R, T

def load_scaled_calibration_parameters(json_file_path):
    """
    Reads scaled camera calibration parameters from a JSON file and returns them as NumPy arrays.

    Args:
        json_file_path (str): The full path to the JSON file containing the parameters.

    Returns:
        dict or None: A dictionary containing the loaded parameters as NumPy arrays,
                      or None if an error occurs during loading.
                      The dictionary will contain keys:
                      'K1_scaled', 'D1_scaled', 'K2_scaled', 'D2_scaled',
                      'R_scaled', 'T_scaled', 'image_size_resized', 'scale_factor_applied'.
    """
    
    

    loaded_params = None
    try:
        streamlit_json_file = False
        if isinstance(json_file_path, str):
            
            with open(json_file_path, 'r') as f:
                loaded_data = json.load(f)
        
        elif hasattr(json_file_path, 'read'):
            streamlit_json_file = True
            loaded_data = json.load(json_file_path)
            # loaded_data = json.load(json_file_path)

        # Extract the required variables, converting lists back to numpy arrays
        K1_scaled = np.array(loaded_data["camera_matrix_1"])
        D1_scaled = np.array(loaded_data["dist_coeff_1"])
        K2_scaled = np.array(loaded_data["camera_matrix_2"])
        D2_scaled = np.array(loaded_data["dist_coeff_2"])
        R_scaled = np.array(loaded_data["Rot_mat"])
        T_scaled = np.array(loaded_data["Trans_vect"])

        # These might be lists/tuples or basic types, no need for np.array conversion
        image_size_resized = loaded_data.get("image_size_resized")
        scale_factor_applied = loaded_data.get("scale_factor_applied")
        image_size_actual = loaded_data.get("image_size_actual")

        loaded_params = {
            "K1": K1_scaled,
            "D1": D1_scaled,
            "K2": K2_scaled,
            "D2": D2_scaled,
            "R": R_scaled,
            "T": T_scaled,
            "image_size_actual": image_size_actual,
            "image_size_resized": image_size_resized,
            "scale_factor_applied": scale_factor_applied
        }

        if streamlit_json_file:
            print(f"Successfully loaded scaled calibration parameters from: {json_file_path.name}")
        else:
            print(f"Successfully loaded scaled calibration parameters from: {json_file_path}")


    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file_path}'. Check file integrity.")
    except KeyError as e:
        print(f"Error: Missing expected key '{e}' in the JSON data. File format might be incorrect.")
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}")

    return loaded_params

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8, force_square=False):
        self.ht, self.wd = dims[-2:]
        if force_square:
          max_side = max(self.ht, self.wd)
          pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
          pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
          pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
          pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def disparity_to_point_cloud(P1, P2, disp_l):
    cam_pts_l, cam_pts_r = [], []
    for i in range(disp_l.shape[0]):
        for j in range(disp_l.shape[1]):
            if disp_l[i, j] != 0:
                cam_pts_l.append([j, i])
                cam_pts_r.append([j + disp_l[i, j], i])

    cam_pts_l, cam_pts_r = np.array(cam_pts_l)[:, np.newaxis, :], np.array(cam_pts_r)[:, np.newaxis, :]
    pts4D = cv2.triangulatePoints(P1, P2, np.float32(cam_pts_l), np.float32(cam_pts_r)).T
    pts3D = pts4D[:, :3] / pts4D[:, -1:]

    # Save point cloud to PLY file
    return numpy_to_o3d(pts3D)

def numpy_to_o3d(pcd_np):
    valid_ids = (~np.isnan(pcd_np).any(axis=1)) & (~np.isinf(pcd_np).any(axis=1))
    valid_pcd = pcd_np[valid_ids]
    print('There are {} points'.format(valid_pcd.shape[0]))
    tmp = o3dg.PointCloud()
    tmp.points = o3d.utility.Vector3dVector(valid_pcd)
    return tmp

def vis_disparity(disp, min_val=None, max_val=None, invalid_thres=np.inf, color_map=cv2.COLORMAP_TURBO, cmap=None, other_output={}):
  """
  @disp: np array (H,W)
  @invalid_thres: > thres is invalid
  """
  disp = disp.copy()
  H,W = disp.shape[:2]
  invalid_mask = disp>=invalid_thres
  if (invalid_mask==0).sum()==0:
    other_output['min_val'] = None
    other_output['max_val'] = None
    return np.zeros((H,W,3))
  if min_val is None:
    min_val = disp[invalid_mask==0].min()
  if max_val is None:
    max_val = disp[invalid_mask==0].max()
  other_output['min_val'] = min_val
  other_output['max_val'] = max_val
  vis = ((disp-min_val)/(max_val-min_val)).clip(0,1) * 255
  if cmap is None:
    vis = cv2.applyColorMap(vis.clip(0, 255).astype(np.uint8), color_map)[...,::-1]
  else:
    vis = cmap(vis.astype(np.uint8))[...,:3]*255
  if invalid_mask.any():
    vis[invalid_mask] = 0
  return vis.astype(np.uint8)

def generate_rectify_data(M1, M2, R, T, d1, d2, size):
    R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(M1, d1, M2, d2, size, R, T, alpha=0, flags=cv2.CALIB_ZERO_TANGENT_DIST)
    map1x, map1y = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, P1, P2, Q

def rectify(img, map_x, map_y):
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

def disparity_to_point_cloud(P1, P2, disp_l):
    cam_pts_l, cam_pts_r = [], []
    for i in range(disp_l.shape[0]):
        for j in range(disp_l.shape[1]):
            if disp_l[i, j] != 0:
                cam_pts_l.append([j, i])
                cam_pts_r.append([j + disp_l[i, j], i])

    cam_pts_l, cam_pts_r = np.array(cam_pts_l)[:, np.newaxis, :], np.array(cam_pts_r)[:, np.newaxis, :]
    pts4D = cv2.triangulatePoints(P1, P2, np.float32(cam_pts_l), np.float32(cam_pts_r)).T
    pts3D = pts4D[:, :3] / pts4D[:, -1:]

    # Save point cloud to PLY file
    return numpy_to_o3d(pts3D)

def is_colab():
  return 'google.colab' in sys.modules

def show_image(image, title="Image"):
    """
    Displays an image using OpenCV. Automatically adapts for Google Colab
    or local environments using the is_colab() helper.

    Args:
        image (numpy.ndarray): The image to display (a NumPy array).
        title (str, optional): The title for the display window (used in local environments).
                               Defaults to "Image".
    """
    if is_colab():
        print(f"Displaying image in Colab output (title '{title}' ignored).")
        # For Colab, import cv2_imshow from google.colab.patches inside the function
        # to ensure it's only attempted when in Colab.
        from google.colab.patches import cv2_imshow
        cv2_imshow(image)
    else:
        print(f"Displaying image in a new window titled '{title}'.")
        # Standard OpenCV display for local environments
        cv2.imshow(title, image)
        cv2.waitKey(0) # Wait indefinitely for a key press
        cv2.destroyAllWindows() # Close all OpenCV windows

def show_contour_map(disp):
    # Assuming 'disp' is your disparity map as a NumPy array
    X, Y = np.meshgrid(np.arange(disp.shape[1]), np.arange(disp.shape[0]))

    fig = go.Figure(data=[go.Surface(z=np.log10(disp), x=X, y=Y)])

    fig.update_layout(title='3D Contour Plot of Disparity Map',
                      scene=dict(
                          xaxis_title='X',
                          yaxis_title='Y',
                          zaxis_title='Disparity'),
                      autosize=True,
                      margin=dict(l=65, r=50, b=65, t=90))

    fig.show(renderer="browser")

def resize_2D_data(array, scale_factor, interpolation_order='cubic'):
    """
    Resizes an N-dimensional NumPy array using scipy.ndimage.zoom().
    Useful for rescaling 2D/3D data like heatmaps, matrices, volumes.

    Args:
        array (numpy.ndarray): Input NumPy array (2D, 3D, etc.).
        scale_factor (float or tuple): Scaling factor(s) for each dimension.
                                       - If float: Applies same scale to all dimensions.
                                       - If tuple: Should match number of dimensions.
        interpolation_order (str or int): Interpolation method:
                                          'nearest' = 0,
                                          'linear'  = 1,
                                          'cubic'   = 3 (default),
                                          'quintic' = 5.
                                          Can also pass the int directly.

    Returns:
        numpy.ndarray: The resized array.
    """

    # Mapping string to scipy interpolation order
    interpolation_map = {
        'nearest': 0,
        'linear': 1,
        'cubic': 3,
        'quintic': 5
    }

    if isinstance(interpolation_order, str):
        if interpolation_order not in interpolation_map:
            raise ValueError(f"Invalid interpolation_order: '{interpolation_order}'. "
                             f"Choose from {list(interpolation_map.keys())}")
        order = interpolation_map[interpolation_order]
    elif isinstance(interpolation_order, int):
        order = interpolation_order
    else:
        raise TypeError("interpolation_order must be a string or an integer.")

    # Handle scale factor
    if isinstance(scale_factor, (float, int)):
        zoom_factors = [scale_factor] * array.ndim
    elif isinstance(scale_factor, tuple) and len(scale_factor) == array.ndim:
        zoom_factors = scale_factor
    else:
        raise ValueError("scale_factor must be a float/int or a tuple matching array dimensions.")

    # Apply zoom
    resized = zoom(array, zoom=scale_factor, order=order)

    return resized

def save_point_cloud(point_cloud, output_base_dir):
    try:

        if not os.path.exists(output_base_dir):
            os.makedirs(output_base_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        outpath = os.path.join(output_base_dir, f'cloud_{timestamp}.ply')
        o3d.io.write_point_cloud(outpath, point_cloud)
        print(f"{bcolors.OKGREEN} Point cloud saved to {outpath} {bcolors.ENDC}")
        return outpath
    except Exception as e:
        print(f"Error saving point cloud: {e}")
        raise e
    
        return None

def load_rectification_artifacts(json_file_path):
    """
    Load rectification artifacts (P1 and P2 matrices) from a JSON file.

    Args:
        json_file_path (str): Path to the JSON file containing rectification artifacts.

    Returns:
        tuple: A tuple containing P1 and P2 as numpy arrays.
    """
    try:
        with open(json_file_path, "r") as f:
            artifacts = json.load(f)
        P1 = np.array(artifacts["P1"])
        P2 = np.array(artifacts["P2"])
        return P1, P2
    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file '{json_file_path}'. Check file integrity.")
    except KeyError as e:
        print(f"Error: Missing expected key '{e}' in the JSON data. File format might be incorrect.")
    except Exception as e:
        print(f"An unexpected error occurred while reading the JSON file: {e}")
        return None, None