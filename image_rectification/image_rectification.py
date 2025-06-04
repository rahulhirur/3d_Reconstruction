import argparse
import os
import json
from rectification_utils import stereo_rectify

def main():
    """
    Main function to handle command line arguments and call the stereo_rectify function.
    """
    ##############################################################################
    # Argument parser setup
    ##############################################################################
    # This sets up the command line argument parser to accept various parameters
    # for the image rectification process. It includes paths for input images, calibration files,
    # output directories, and options for resizing, perspective usage, and saving data.
    parser = argparse.ArgumentParser(description="Image Rectification Tool")
    parser.add_argument("--image1_path", type=str, required=True, help="Path to the left image")
    parser.add_argument("--image2_path", type=str, required=True, help="Path to the right image")
    parser.add_argument("--calib_file", type=str, required=True, help="Path to the calibration file (JSON format)")
    parser.add_argument("--output_base_dir", type=str, default="rectified_images", help="Base directory to save rectified images")
    parser.add_argument("--new_scale_factor", type=float, default=0.25, help="Scale factor for resizing images")
    parser.add_argument("--cam1_perspective", action='store_true', help="Use camera 1 perspective for rectification")
    parser.add_argument("--save_data", action='store_true', help="Save rectified images and data")
    parser.add_argument("--show_images", action='store_true', help="Display rectified images")

    args = parser.parse_args()

    stereo_rectify(
        image1_path=args.image1_path,
        image2_path=args.image2_path,
        calib_file=args.calib_file,
        output_base_dir=args.output_base_dir,
        new_scale_factor=args.new_scale_factor,
        cam1_perspective=args.cam1_perspective,
        save_data=args.save_data,
        show_images=args.show_images
    )

    # save args log file if save_data is True in output_base_dir
    if args.save_data:
        log_file_path = os.path.join(args.output_base_dir, "args_log.json")
        with open(log_file_path, 'w') as log_file:
            json.dump(vars(args), log_file, indent=4)
        print(f"Arguments saved to {log_file_path}")

if __name__ == "__main__":
    main()