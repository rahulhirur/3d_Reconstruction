import cv2
import json
import numpy as np
import os
import yaml
import re
import sys

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
        print("Press any key on the keyboard to close the window.")
        # Standard OpenCV display for local environments
        cv2.imshow(title, image)
        cv2.waitKey(0) # Wait indefinitely for a key press
        cv2.destroyAllWindows() # Close all OpenCV windows

class YamlCameraCalibration:
    def __init__(self, yaml_file_path):

        self.data = cv2.FileStorage(yaml_file_path, cv2.FILE_STORAGE_READ)
        if not self.data.isOpened():
            raise IOError(f"Could not open YAML file: {yaml_file_path}")

    def load_camera_calibration(self, camera_index, scale_factor=1.0):
        camera_key = f"camera_matrix_{camera_index + 1}"
        dist_key = f"dist_coeff_{camera_index + 1}"

        # Load original intrinsic matrix
        K_original_data = self.data.getNode(camera_key).mat()
        if K_original_data is None:
            raise ValueError(f"Could not find camera matrix for {camera_key} in YAML.")
        K_original = K_original_data.reshape((3, 3))

        # Apply scaling to the intrinsic matrix
        K_scaled = K_original.copy()
        K_scaled[0, 0] *= scale_factor  # fx
        K_scaled[1, 1] *= scale_factor  # fy
        K_scaled[0, 2] *= scale_factor  # cx
        K_scaled[1, 2] *= scale_factor  # cy

        # Load distortion coefficients (remain unchanged by image scaling)
        D_data = self.data.getNode(dist_key).mat()
        if D_data is None:
            raise ValueError(f"Could not find distortion coefficients for {dist_key} in YAML.")
        D = D_data.flatten()

        return K_scaled, D

    def load_stereo_calibration(self):

        R_data = self.data.getNode('Rot_mat').mat()
        if R_data is None:
            raise ValueError("Could not find 'Rot_mat' in YAML.")
        R = np.array(R_data).reshape((3, 3))

        T_data = self.data.getNode('Trans_vect').mat()
        if T_data is None:
            raise ValueError("Could not find 'Trans_vect' in YAML.")
        T = np.array(T_data).reshape((3, 1))

        return R, T

    def get_all_calibration(self, scale_factor=1.0):

        K1, D1 = self.load_camera_calibration(0, scale_factor)
        K2, D2 = self.load_camera_calibration(1, scale_factor)
        R, T = self.load_stereo_calibration() # R and T are not scaled
        return K1, D1, K2, D2, R, T

    def close(self):
        self.data.release()

class JsonCameraCalibration:
    def __init__(self, json_file_path):

        try:
            with open(json_file_path, 'r') as f:
                self.calibration_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from file: {json_file_path}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while loading JSON: {e}")

    def load_camera_calibration(self, camera_index, scale_factor=1.0):

        try:
            camera_data = self.calibration_data["calibration"]["cameras"][camera_index]
            parameters = camera_data["model"]["ptr_wrapper"]["data"]["parameters"]
        except KeyError as e:
            raise KeyError(f"Missing key in JSON structure for camera {camera_index}: {e}")

        # Original intrinsic parameters
        fx_original = parameters["f"]["val"]
        fy_original = fx_original / parameters["ar"]["val"] # Assuming ar is fx/fy
        cx_original = parameters["cx"]["val"]
        cy_original = parameters["cy"]["val"]

        # Apply scaling to intrinsic parameters
        fx_scaled = fx_original * scale_factor
        fy_scaled = fy_original * scale_factor
        cx_scaled = cx_original * scale_factor
        cy_scaled = cy_original * scale_factor

        K_scaled = np.array([[fx_scaled, 0, cx_scaled],
                             [0, fy_scaled, cy_scaled],
                             [0, 0, 1]], dtype=np.float64)

        # Distortion coefficients (remain unchanged by image scaling)
        k1 = parameters["k1"]["val"]
        k2 = parameters["k2"]["val"]
        k3 = parameters["k3"]["val"]
        p1 = parameters["p1"]["val"]
        p2 = parameters["p2"]["val"]
        D = np.array([k1, k2, p1, p2, k3], dtype=np.float64)

        return K_scaled, D

    def load_stereo_calibration(self, camera_index=1):

        try:
            # Assuming 'transform' describes the pose of camera_index relative to camera 0
            transform_data = self.calibration_data["calibration"]["cameras"][camera_index]["transform"]

            # Convert Rodrigues vector to Rotation Matrix
            rx = transform_data["rotation"]["rx"]
            ry = transform_data["rotation"]["ry"]
            rz = transform_data["rotation"]["rz"]
            R_vec = np.array([rx, ry, rz], dtype=np.float64)
            R, _ = cv2.Rodrigues(R_vec) # cv2.Rodrigues returns R and Jacobian, we only need R

            # Extract Translation Vector
            tx = transform_data["translation"]["x"]
            ty = transform_data["translation"]["y"]
            tz = transform_data["translation"]["z"]
            T = np.array([[tx], [ty], [tz]], dtype=np.float64)

        except KeyError as e:
            raise KeyError(f"Missing key in JSON structure for stereo transformation: {e}")
        except Exception as e:
            raise Exception(f"An unexpected error occurred while loading stereo parameters: {e}")

        return R, T

    def get_all_calibration(self, scale_factor=1.0):

        K1, D1 = self.load_camera_calibration(0, scale_factor)
        K2, D2 = self.load_camera_calibration(1, scale_factor)
        R, T = self.load_stereo_calibration(camera_index=1)

        return K1, D1, K2, D2, R, T

    def close(self):
        pass # No explicit resource to release for json.load

class CalibrationLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        ext = os.path.splitext(file_path)[1].lower()

        if ext in ['.json']:
            self.loader = JsonCameraCalibration(file_path)
        elif ext in ['.yaml', '.yml']:
            self.loader = YamlCameraCalibration(file_path)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def get_all_calibration(self, scale_factor):
        return self.loader.get_all_calibration(scale_factor)

    def close(self):
        if hasattr(self.loader, 'close'):
            self.loader.close()

# Get Rectification map
def generate_rectify_data(M1, M2, R, T, d1, d2, size):
    R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(M1, d1, M2, d2, size, R, T, alpha=0, flags=cv2.CALIB_ZERO_TANGENT_DIST)
    map1x, map1y = cv2.initUndistortRectifyMap(M1, d1, R1, P1, size, cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(M2, d2, R2, P2, size, cv2.CV_32FC1)
    return map1x, map1y, map2x, map2y, P1, P2, Q

def rectify(img, map_x, map_y):
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)

def show_stereo_images(img_1, img_2):
    # Create an image for the line with the same height as the resized images
    line_img = np.full((img_1.shape[0], 2, 3), (0, 255, 255), dtype=np.uint8)
    show_image(np.concatenate([cv2.cvtColor(img_1, cv2.COLOR_GRAY2BGR),line_img, cv2.cvtColor(img_2, cv2.COLOR_GRAY2BGR)], axis=1))

def resize_image(image, scale_factor):
    """
    Resizes an image based on a given scale factor.

    Args:
        image (np.ndarray): The input image (OpenCV format).
        scale_factor (float): The factor by which to scale the image.

    Returns:
        np.ndarray: The resized image.
        tuple: The new dimensions of the resized image (width, height).
    """
    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive.")

    # Get original dimensions
    original_height, original_width = image.shape[:2]

    # Calculate new dimensions
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    return resized_image, (new_width, new_height)

def save_images(rectified_img1, rectified_img2, output_base_dir="/output"):
    """
    Saves two rectified images to a specified output directory.

    Args:
        rectified_img1 (numpy.ndarray): The first rectified image (e.g., left camera).
        rectified_img2 (numpy.ndarray): The second rectified image (e.g., right camera).
        output_base_dir (str, optional): The base directory where images will be saved.
                                        Defaults to "/content/rectified_images"
                                        (common for Colab).
    Returns:
        tuple: A tuple containing the paths of the two saved images (path1, path2).
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_base_dir}")
    
    # Define the filenames for the rectified images
    rectified_img1_path = os.path.join(output_base_dir, "rectified_left.png")
    rectified_img2_path = os.path.join(output_base_dir, "rectified_right.png")

    # Save the rectified images
    cv2.imwrite(rectified_img1_path, rectified_img1)
    cv2.imwrite(rectified_img2_path, rectified_img2)

    print(f"Rectified images saved:")
    print(f"  - Left: {rectified_img1_path}")
    print(f"  - Right: {rectified_img2_path}")

    return rectified_img1_path, rectified_img2_path

def save_scaled_calibration_parameters(K1_scaled, D1_scaled, K2_scaled, D2_scaled,
                                       R_scaled, T_scaled, img_size, new_scale_factor,
                                       output_dir="/output"):
    """
    Saves scaled camera calibration parameters to a JSON file.

    Args:
        K1_scaled (numpy.ndarray): Scaled camera matrix for camera 1.
        D1_scaled (numpy.ndarray): Scaled distortion coefficients for camera 1.
        K2_scaled (numpy.ndarray): Scaled camera matrix for camera 2.
        D2_scaled (numpy.ndarray): Scaled distortion coefficients for camera 2.
        R_scaled (numpy.ndarray): Scaled rotation matrix (between cameras).
        T_scaled (numpy.ndarray): Scaled translation vector (between cameras).
        img_size (tuple): The resized image dimensions (width, height) used for scaling.
        new_scale_factor (float): The scale factor applied to the original parameters.
        output_dir (str, optional): The directory to save the JSON file. Defaults to "/content".

    Returns:
        str or None: The path to the saved JSON file if successful, None otherwise.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    # Convert NumPy arrays to lists for JSON serialization
    K1_scaled_list = K1_scaled.tolist()
    D1_scaled_list = D1_scaled.tolist()
    K2_scaled_list = K2_scaled.tolist()
    D2_scaled_list = D2_scaled.tolist()
    R_scaled_list = R_scaled.tolist()
    T_scaled_list = T_scaled.tolist()

    # Create a dictionary to hold the scaled calibration parameters
    scaled_calibration_data = {
        "camera_matrix_1": K1_scaled_list,
        "dist_coeff_1": D1_scaled_list,
        "camera_matrix_2": K2_scaled_list,
        "dist_coeff_2": D2_scaled_list,
        "Rot_mat": R_scaled_list,
        "Trans_vect": T_scaled_list,
        "image_size_resized": img_size,
        "image_size_actual": (np.array(img_size)*(1/0.25)).tolist(),
        "scale_factor_applied": new_scale_factor,
        "baseline_distance": np.linalg.norm(T_scaled)  # Distance between cameras

    }

    # Define the output JSON file path
    scaled_calibration_output_path = os.path.join(output_dir, "scaled_calibration_parameters.json")

    # Save the scaled calibration parameters to a JSON file
    try:
        with open(scaled_calibration_output_path, 'w') as f:
            json.dump(scaled_calibration_data, f, indent=4) # indent=4 makes the JSON readable
        print(f"Scaled calibration parameters saved to: {scaled_calibration_output_path}")
        return scaled_calibration_output_path
    except Exception as e:
        print(f"Error saving scaled calibration parameters to JSON: {e}")
        return None

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
        with open(json_file_path, 'r') as f:
            loaded_data = json.load(f)

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

        loaded_params = {
            "K1": K1_scaled,
            "D1": D1_scaled,
            "K2": K2_scaled,
            "D2": D2_scaled,
            "R": R_scaled,
            "T": T_scaled,
            "image_size_resized": image_size_resized,
            "scale_factor_applied": scale_factor_applied
        }
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


def update_calibration_data_scaled(K1_scaled, D1_scaled, K2_scaled, D2_scaled, R_scaled, T_scaled, img_size, new_scale_factor, html_path="/assets/template.html", output_dir="/output"):
    """
    Updates the calibrationData object in an HTML file with new scaled camera parameters.

    Parameters:
        html_path (str): Path to the input HTML file.
        output_path (str): Path to save the updated HTML file.
        K1_scaled, K2_scaled (list of list): 3x3 camera matrix.
        D1_scaled, D2_scaled (list): Distortion coefficients.
        R_scaled (list of list): 3x3 rotation matrix.
        T_scaled (list of list): 3x1 translation vector.
        img_size (list): New image size [width, height].
        new_scale_factor (float): New scale factor.
    """
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Extract calibrationData JS object using regex
    pattern = r"(const calibrationData\s*=\s*)(\{.*?\})(\s*;)"
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        raise ValueError("calibrationData block not found in the HTML.")

    prefix, js_object_str, suffix = match.groups()

    # Convert JS object to JSON-compatible string
    json_compatible = re.sub(r'(\w+):', r'"\1":', js_object_str)
    json_compatible = json_compatible.replace(";", "")
    data = json.loads(json_compatible)

    # Convert NumPy arrays to lists for JSON serialization
    K1_scaled_list = K1_scaled.tolist()
    D1_scaled_list = D1_scaled.tolist()
    K2_scaled_list = K2_scaled.tolist()
    D2_scaled_list = D2_scaled.tolist()
    R_scaled_list = R_scaled.tolist()
    T_scaled_list = T_scaled.tolist()

    # Apply updates
    data["camera_matrix_1"] = K1_scaled_list
    data["dist_coeff_1"] = D1_scaled_list
    data["camera_matrix_2"] = K2_scaled_list
    data["dist_coeff_2"] = D2_scaled_list
    data["Rot_mat"] = R_scaled_list
    data["Trans_vect"] = T_scaled_list
    data["image_size_resized"] = img_size
    data["scale_factor_applied"] = new_scale_factor
    data["image_size_original"] = (np.array(img_size) * (1 / new_scale_factor)).tolist()  # Original size
    data["baseline_distance"] = np.linalg.norm(T_scaled_list)  # Distance between cameras L2 Distance

    # Convert back to JS-style object string
    new_js = json.dumps(data, indent=4)
    new_js = re.sub(r'"(\w+)":', r'\1:', new_js)

    updated_html = html[:match.start()] + prefix + new_js + suffix + html[match.end():]

    with open(os.path.join(output_dir, "Stereo_Calibration_Parameters.html"), "w", encoding="utf-8") as f:
        f.write(updated_html)

    print(f"calibrationData updated and saved to '{output_dir}'")



