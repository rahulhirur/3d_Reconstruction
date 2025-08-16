import argparse
from pipeline_utils import run_reconstruction

def main():
    parser = argparse.ArgumentParser(description="3D Reconstruction Pipeline")
    parser.add_argument("--left_img", type=str, required=True, help="Path to the left image")
    parser.add_argument("--right_img", type=str, required=True, help="Path to the right image")
    parser.add_argument("--params_file", type=str, required=True, help="Path to the calibration parameters JSON file")
    parser.add_argument("--ground_truth_pcd", type=str, help="Path to the ground truth PCD file for comparison (optional)")
    parser.add_argument("--output_folder", type=str, required=True, help="Base directory to save output data")
    parser.add_argument("--scale_factor", type=float, default=0.25, help="Scale factor for resizing images and calibration parameters")
    parser.add_argument("--cam1_perspective", action='store_false', help="Dont use the first camera as the reference for rectification")
    parser.add_argument("--save_data", action='store_false', help="Save the rectified images and scaled calibration parameters")
    parser.add_argument("--show_images", action='store_true', help="Display the rectified images")

    args = parser.parse_args()

    run_reconstruction(
        left_img_path=args.left_img,
        right_img_path=args.right_img,
        params_json_path=args.params_file,
        ground_truth_pcd_path=args.ground_truth_pcd,
        out_dir=args.output_folder,
        scale_factor=args.scale_factor,
        cam1_perspective=args.cam1_perspective,
        save_data=args.save_data,
        show_images=args.show_images
    )

if __name__ == "__main__":
    main()