import json
import os
import argparse

# Import the necessary functions from the correct subdirectories

from image_rectification.rectification_utils import stereo_rectify
from image_rectification.functions import suggest_next_folder_name

from disparity_calculator.disparity_utils import calc_disparity

from ply_generation.ply_generation_utils import generate_pcd

from pcd_comparison.pcd_comparison_utils import compare_pcd


def run_reconstruction(left_img_path, right_img_path, params_json_path, ground_truth_pcd_path,  out_dir, scale_factor=0.25, cam1_perspective=True, save_data=True, show_images=False):
    """
    Runs the full 3D reconstruction pipeline from start to finish.
    """

    print("Starting pipeline...")
    
    output_dir = suggest_next_folder_name(out_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")


    # 1. Load prerequisites
    print("Step 0: Loading prerequisites...")
    #Create a folder for the rectified images
    
    output_rect_dir = suggest_next_folder_name(output_dir, prefix="rectified_image_batch_")
    os.makedirs(output_rect_dir, exist_ok=True)
    print(f"Rectified images will be saved in: {output_rect_dir}")

    # 2. Image rectification
    print("Step 1: Rectifying images...")

    stereo_rectify(
        image1_path=left_img_path,
        image2_path=right_img_path,
        calib_file=params_json_path,  # Pass the calibration file path
        output_base_dir=output_rect_dir,
        new_scale_factor=scale_factor,
        cam1_perspective=cam1_perspective,
        save_data=save_data,
        show_images=show_images
    )
    

    # 3. Disparity calculation
    print("Step 2: Calculating disparity map...")

    rectified_left_path = os.path.join(output_rect_dir, "rectified_image1.png")  # Construct the path
    rectified_right_path = os.path.join(output_rect_dir, "rectified_image2.png")  # Construct the path
    scaled_calib_path = os.path.join(output_rect_dir, "scaled_calibration_parameters.json")  # Path to the scaled calibration file

    output_disp_dir = suggest_next_folder_name(output_rect_dir, prefix="disparity_batch_")

    calc_disparity(rectified_left_path, rectified_right_path, output_disp_dir)

    disp_path = os.path.join(output_disp_dir, "disp.npy")  # Path to the disparity file

    # 4. PLY generation
    print("Step 3: Generating PLY file...")

    output_cloud_dir = suggest_next_folder_name(output_rect_dir, prefix="cloud_batch_")

    generate_pcd(
        disp_path=disp_path,
        img_path=rectified_left_path,
        calib_file=scaled_calib_path,
        output_base_dir=output_cloud_dir,
        threshold_value=450,
        resize_disparity=True, 
        visualize_disparity=False,
        name_timestamp=False
    )


    ply_file_path = os.path.join(output_cloud_dir, "point_cloud.ply")  # Path to the generated PLY file

    # 5. PCD comparison (optional)
    if ground_truth_pcd_path is not None:
        print("Step 4: Comparing point clouds...")
        trans_init_method = "identity"  # or "random" based on your requirement
        threshold = 0.05  # Set a threshold for comparison, adjust as needed
        max_iter = 200
        compare_pcd(ply_file_path, ground_truth_pcd_path, threshold, trans_init_method, max_iter, output_dir=None)

    print("Pipeline completed successfully! ✨")
