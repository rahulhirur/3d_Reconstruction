from functions import *

def generate_pcd(
        disp_path: str,
        img_path: str,
        calib_file: str,
        output_base_dir: str,
        threshold_value: int = 480,
        resize_disparity: bool = True,
        visualize_disparity: bool = True
):

    disp = np.load(disp_path)

    img = cv2.imread(img_path)
    img_size= img.shape[:2]

    # Apply the threshold
    disp[disp < threshold_value] = 0
    
    
    if resize_disparity:
        scale_factor = img_size[1] / disp.shape[1]
        if scale_factor != 1:
            disp = resize_2D_data(disp, scale_factor, interpolation_order='cubic')
            print(disp.shape)
    
    vis = vis_disparity(disp)
    

    calib_parameters = load_scaled_calibration_parameters(calib_file)
    map_x_l, map_y_l, map_x_r, map_y_r, P1, P2, Q = generate_rectify_data(calib_parameters["K1"], calib_parameters["K2"], calib_parameters["R"], calib_parameters["T"], calib_parameters["D1"], calib_parameters["D2"], disp.shape[:2])
    
    point_cloud = disparity_to_point_cloud(P1, P2, disp)
    
    cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
    point_cloud = point_cloud.select_by_index(ind)

    save_point_cloud(point_cloud, output_base_dir)

    if visualize_disparity:
        show_image(vis, title="Disparity Map")

