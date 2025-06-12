import argparse
import os
import json
from pcd_generation_utils import generate_pcd

def main():
    """
    Main function to handle command line arguments and call the generate_pcd function.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Point cloud Generation Tool")
    parser.add_argument("--disp_path", type=str, required=True, help="Path to disparity file (Numpy format)")
    parser.add_argument("--img_path", type=str, required=True, help="Path to the rectified image")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to the calibration file (JSON format)")
    parser.add_argument("--output_base_dir", type=str, help="Base directory to save point cloud data")
    parser.add_argument("--threshold_value", type=float, default=400, help="Threshold factor to filter disparity values")
    parser.add_argument("--resize_disparity", action='store_true', help="Resize disparity based on image size")
    parser.add_argument("--visualize_disparity", action='store_true', help="Visaulize the disparity map")
    
    args = parser.parse_args()

    generate_pcd(disp_path= args.disp_path,
            img_path = args.img_path,
            calib_file= args.calib_file,
            output_base_dir= args.output_base_dir, 
            threshold_value = args.threshold_value,
            resize_disparity = args.resize_disparity,
            visualize_disparity = args.visualize_disparity)
    
    # save args log file if output_base_dir is provided
    if args.output_base_dir:
        log_file_path = os.path.join(args.output_base_dir, "pcd_args_log.json")
        with open(log_file_path, 'w') as log_file:
            json.dump(vars(args), log_file, indent=4)
        print(f"Arguments saved to {log_file_path}")


if __name__ == "__main__":
    main()
