import argparse
import os
import json
from pcd_comparison_utils import compare_pcd

def main():
    """
    Main function to handle command line arguments and call the compare_pcd function.
    """
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Point Cloud Comparison Tool")
    parser.add_argument("--source_file", type=str, required=True, help="Path to the source point cloud file")
    parser.add_argument("--target_file", type=str, required=True, help="Path to the target point cloud file")
    parser.add_argument("--threshold", type=float, default=0.05, help="Distance threshold for ICP registration")
    parser.add_argument("--trans_init_method", type=str, default="identity", help="Method for generating initial transformation matrix")
    parser.add_argument("--max_iter", type=int, default=30, help="Maximum number of iterations for ICP")
    parser.add_argument("--output_dir", type=str, default = None, help="Directory to save comparison results")
    
    args = parser.parse_args()

    compare_pcd(source_file=args.source_file,
                target_file=args.target_file,
                threshold=args.threshold,
                trans_init_method=args.trans_init_method,
                max_iter=args.max_iter,
                output_dir=args.output_dir)
    
if __name__ == "__main__":
    main()