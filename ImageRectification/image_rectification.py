from functions import *

# Read stereo images
image1_path = "/home/hirur/MasterThesis/3d_Reconstruction/images/batch_1/left.png"
image2_path = "/home/hirur/MasterThesis/3d_Reconstruction/images/batch_1/right.png"

#read images
img_1 = cv2.imread(image1_path)
img_2 = cv2.imread(image2_path)

# Convert images to single channel (grayscale)
# img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
# img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)


new_scale_factor = 0.25
# Read JSON calibration file
calib_file = "/home/hirur/MasterThesis/3d_Reconstruction/images/batch_1/calibration_file.json"
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

# Generate rectification maps using the scaled parameters and new image size
map1x, map1y, map2x, map2y, P1, P2, Q = generate_rectify_data(K1_scaled, K2_scaled,  R_scaled, T_scaled, D1_scaled, D2_scaled, img_size)

# Rectify the resized images
rectified_img1 = rectify(img_1_resized, map1x, map1y)
rectified_img2 = rectify(img_2_resized, map2x, map2y)

# Save rectified images and scaled calibration parameters
save_images(rectified_img1, rectified_img2, output_base_dir="rectified_images/batch_2")
save_scaled_calibration_parameters(K1_scaled, D1_scaled, K2_scaled, D2_scaled,
                                       R_scaled, T_scaled, img_size, new_scale_factor,
                                       output_dir="rectified_images/batch_2")
# Show rectified stereo images
# Create an image for the line with the same height as the resized images
line_img = np.full((rectified_img1.shape[0], 2, 3), (0, 255, 255), dtype=np.uint8)
if len(rectified_img1.shape) == 2:  # If grayscale, convert to BGR for display
    rectified_img1 = cv2.cvtColor(rectified_img1, cv2.COLOR_GRAY2BGR)
    rectified_img2 = cv2.cvtColor(rectified_img2, cv2.COLOR_GRAY2BGR)

show_image(np.concatenate([rectified_img1,line_img,rectified_img2], axis=1))

