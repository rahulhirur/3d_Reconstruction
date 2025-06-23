import argparse
import os
import json

from disparity_utils import calc_disparity

def main():
    """
    Main function to handle command line arguments and call the calc_disparity function.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Disparity Calculation Tool")
    parser.add_argument("--img0", type=str, required=True, help="Path to the first image (left)")
    parser.add_argument("--img1", type=str, required=True, help="Path to the second image (right)")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save the disparity output")
    parser.add_argument("--hiera", action='store_true', help="Use hierarchical mode for disparity calculation")
    #scale should be a float
    parser.add_argument("--scale", type=float, default=1.0, help="Scale factor for images")

    args = parser.parse_args()

    calc_disparity(args.img0, args.img1, args.out_dir, args.scale)


    # save args log file
    log_file_path = os.path.join(args.out_dir, "disparity_args_log.json")
    with open(log_file_path, 'w') as log_file:
        json.dump(vars(args), log_file, indent=4)
    print(f"Arguments saved to {log_file_path}")

if __name__ == "__main__":
    main()

