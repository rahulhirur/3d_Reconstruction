from functions import *

def stereo_rectify(
    image1_path: str,
    image2_path: str,
    calib_file: str,
    output_base_dir: str = "rectified_images",
    new_scale_factor: float = 0.25,
    cam1_perspective: bool = False,
    save_data: bool = False,
    show_images: bool = True
):
    """    
    Rectifies stereo images based on calibration data and specified parameters.
    input:
    - image1_path: Path to the first stereo image.
    - image2_path: Path to the second stereo image.
    - calib_file: Path to the JSON calibration file.
    - output_base_dir: Directory to save the rectified images and calibration parameters.
    - new_scale_factor: Scale factor for resizing images and calibration parameters.
    - cam1_perspective: If True, uses the first camera as the reference for rectification.
    - save_data: If True, saves the rectified images and scaled calibration parameters.
    - show_images: If True, displays the rectified images.
    output:
    - Displays the rectified stereo images side by side.
    - Saves the rectified images and scaled calibration parameters if save_data is True.
    - Returns nothing.
    This function reads stereo images, resizes them, retrieves calibration parameters, generates rectification maps,
    rectifies the images, and optionally saves and displays the results.
    It uses the functions defined in the 'functions' module for image processing and calibration handling.
    
    Example usage:
    ImageRectification(
        image1_path="/path/to/left_image.png",
        image2_path="/path/to/right_image.png",
        calib_file="/path/to/calibration_file.json",
        new_scale_factor=0.25,
        cam1_perspective=True,
        save_data=True,
        show_images=True
    )

    Note: Ensure that the paths to the images and calibration file are correct.
    """

    # Read stereo images
    #read images
    img_1 = cv2.imread(image1_path)
    img_2 = cv2.imread(image2_path)

    # Read JSON calibration file
    calib_yaml = CalibrationLoader(calib_file)

    img_1_resized, img_size_resized = resize_image(img_1, new_scale_factor)
    img_2_resized, _ = resize_image(img_2, new_scale_factor) # Use the size from the first image

    print("Resized image 1 size:", img_1_resized.shape[:2])
    print("Resized image 2 size:", img_2_resized.shape[:2])

    # Update img_size to the new size for subsequent steps like rectification
    img_size = img_size_resized # Note: cv2 expects size as (width, height) tuple
    print("Updated img_size for rectification:", img_size)

    # Now you would use img_1_resized, img_2_resized, and the updated img_size
    # along with the scaled calibration matrices (K1, K2 from calib_yaml.get_all_calibration(new_scale_factor))
    # for rectification.
    # Get scaled calibration parameters for the new image size
    K1_scaled, D1_scaled, K2_scaled, D2_scaled, R_scaled, T_scaled = calib_yaml.get_all_calibration(new_scale_factor)

    if cam1_perspective:
        # Generate rectification maps using the scaled parameters and new image size
        map1x, map1y, map2x, map2y, P1, P2, Q = generate_rectify_data(K1_scaled, K2_scaled,  R_scaled, T_scaled, D1_scaled, D2_scaled, img_size)

        # Rectify the resized images
        rectified_img1 = rectify(img_1_resized, map1x, map1y)
        rectified_img2 = rectify(img_2_resized, map2x, map2y)
    else:
        R_2_to_1 = R_scaled.T
        T_2_to_1 = -R_scaled.T @ T_scaled
        # Generate rectify data with right camera as reference
        map_x_new_1, map_y_new_1, map_x_new_2, map_y_new_2, P1_new, P2_new, Q_new = generate_rectify_data(K2_scaled, K1_scaled, R_2_to_1, T_2_to_1, D2_scaled, D1_scaled, img_size)

        rectified_img2 = rectify(img_2_resized, map_x_new_1, map_y_new_1)
        rectified_img1 = rectify(img_1_resized, map_x_new_2, map_y_new_2)

        R_scaled = R_2_to_1
        T_scaled = T_2_to_1

    if save_data:
        
        # Save rectified images and scaled calibration parameters
        save_images(rectified_img1, rectified_img2, output_base_dir=output_base_dir)

        save_scaled_calibration_parameters(K1_scaled, D1_scaled, K2_scaled, D2_scaled,
                                            R_scaled, T_scaled, img_size, new_scale_factor,
                                            output_dir=output_base_dir)
        
        create_calibration_data_report(K1_scaled, D1_scaled, K2_scaled, D2_scaled,
                                        R_scaled, T_scaled, img_size, new_scale_factor, html_path="assets/template.html",
                                        output_dir=output_base_dir)
        
    if show_images:
        
        # Show rectified stereo images
        # Create an image for the line with the same height as the resized images
        line_img = np.full((rectified_img1.shape[0], 2, 3), (0, 255, 255), dtype=np.uint8)
        if len(rectified_img1.shape) == 2:  # If grayscale, convert to BGR for display
            rectified_img1 = cv2.cvtColor(rectified_img1, cv2.COLOR_GRAY2BGR)
            rectified_img2 = cv2.cvtColor(rectified_img2, cv2.COLOR_GRAY2BGR)

        if cam1_perspective:
            show_image(np.concatenate([rectified_img1,line_img,rectified_img2], axis=1))
        else:
            show_image(np.concatenate([rectified_img2,line_img,rectified_img1], axis=1))

