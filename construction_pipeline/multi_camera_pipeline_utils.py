import os
import glob
from construction_pipeline.pipeline_utils import run_reconstruction  # Import the existing function
from image_rectification.functions import suggest_next_folder_name


def process_scenario(scenario_path, ground_truth_pcd_path, out_dir, best_rmse, scale_factor=0.25, cam1_perspective=True, save_data=True, show_images=False, rmse_threshold=0.01):
    """
    Processes all stereo pairs within a scenario folder using the run_reconstruction function.
    Keeps track of the best model (lowest RMSE) and stops processing when RMSE falls below a threshold.
    """

    if not os.path.exists(scenario_path):
        raise ValueError(f"The scenario path does not exist: {scenario_path}")

    camera_pair_folders = sorted(glob.glob(os.path.join(scenario_path, "camera_pair_*")))  # Sort to ensure consistent order

    if not camera_pair_folders:
        raise ValueError(f"No 'camera_pair_*' folders found in the scenario path: {scenario_path}")

    best_output_dir = None

    for camera_pair_folder in camera_pair_folders:
        left_img_path = os.path.join(camera_pair_folder, "left_image.png")
        right_img_path = os.path.join(camera_pair_folder, "right_image.png")
        params_json_path = os.path.join(camera_pair_folder, "params.json")

        if not (os.path.exists(left_img_path) and os.path.exists(right_img_path) and os.path.exists(params_json_path)):
            print(f"Warning: Missing files in {camera_pair_folder}. Skipping this pair.")
            continue

        output_dir = suggest_next_folder_name(out_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Call the existing run_reconstruction function for each pair
        rmse = run_reconstruction(left_img_path, right_img_path, params_json_path, ground_truth_pcd_path, output_dir, scale_factor, cam1_perspective, save_data, show_images)
        print(f"Processed {camera_pair_folder} with RMSE: {rmse}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_output_dir = output_dir

        if rmse < rmse_threshold:  # Use the provided threshold
            print(f"RMSE {rmse} is below the threshold ({rmse_threshold}). Exiting loop.")
            break

    if best_output_dir:
        print(f"\nBest model found in: {best_output_dir} with RMSE: {best_rmse}")
    else:
        print("\nNo model found with RMSE below the threshold.")