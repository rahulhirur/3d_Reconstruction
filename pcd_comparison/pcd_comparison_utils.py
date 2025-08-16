from functions import load_and_preprocess_pcd, add_noise_to_pcd, generate_transformation_init, icp_align_and_compare, save_comparison_results

def compare_pcd(source_file, target_file, threshold, trans_init_method, max_iter, output_dir=None):
    """
    Compare two point clouds by performing ICP registration and returning the transformation matrix.
    
    Args:
        source_file (str): Path to the source point cloud file.
        target_file (str): Path to the target point cloud file.
        threshold (float): Distance threshold for ICP registration
        """
    
    source_pcd = load_and_preprocess_pcd(source_file)
    target_pcd = load_and_preprocess_pcd(target_file)

    # Add noise to the target point cloud
    target_pcd = add_noise_to_pcd(target_pcd, std_dev=0.01)

    # Generate initial transformation matrix
    trans_init_matrix = generate_transformation_init(trans_init_method)

    # Perform ICP registration
    result_transformation, rmse, chamfer, hausdorff = icp_align_and_compare(source_pcd, target_pcd, threshold=threshold, trans_init= trans_init_matrix, max_iter=max_iter)

    print(f"Transformation Matrix:\n{result_transformation}")
    print(f"RMSE: {rmse}, Chamfer Distance: {chamfer}, Hausdorff Distance: {hausdorff}")

    #save data to a json file
    if output_dir:
        save_comparison_results(output_dir, result_transformation, rmse, chamfer, hausdorff)
        print(f"Comparison results saved to {output_dir}")
    
    
    #Create a function of the above code to save json file



