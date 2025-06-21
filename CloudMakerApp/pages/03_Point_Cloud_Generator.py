import streamlit as st
import os
import numpy as np
import open3d as o3d
import plotly.graph_objects as go
import sys

import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# from ply_generation.pcd_generation_utils import generate_pcd
from ply_generation.functions import load_scaled_calibration_parameters, vis_disparity, resize_2D_data, disparity_to_point_cloud, save_point_cloud, generate_rectify_data


#create a buutton to set the threshold value
if 'set_threshold_clicked' not in st.session_state:
    st.session_state['set_threshold_clicked'] = False

# Add session state for threshold value
if 'threshold_value' not in st.session_state:
    st.session_state['threshold_value'] = 480


def visualize_point_cloud(pcd):
    """
    Visualizes a point cloud using Plotly.

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to visualize.
    """
    points = np.asarray(pcd.points)
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=points[:, 2], colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    return fig

st.set_page_config(
    page_title="3D Point Cloud Generator",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("☁️ 3D Point Cloud Generation")
st.markdown("""
Upload the required files to generate a 3D point cloud:
- Disparity file (NumPy format)
- Rectified image
- Calibration file (JSON format)
""")

# File uploaders
#use columns to organize the layout

colId1 = st.columns(2)

disp_path = colId1[0].file_uploader("Upload Disparity File", type="npy")

if disp_path:
    disp = np.load(disp_path)

calib_file = colId1[1].file_uploader("Upload Calibration File", type="json")

if calib_file:
    calib_parameters = load_scaled_calibration_parameters(calib_file)
    img_size= calib_parameters["image_size_actual"]


# Additional parameters
output_base_dir = st.text_input("**Output Directory**", value="output/point_cloud")

coldId2 = st.columns([9,1], vertical_alignment="center")

threshold_value = coldId2[0].slider("**Threshold Value**", min_value=0, max_value=1000, value=st.session_state['threshold_value'], step=10)


if coldId2[1].button("Set Threshold Value", type="primary"):
    st.session_state['set_threshold_clicked'] = True
    st.session_state['threshold_value'] = threshold_value

    if disp_path:
        st.toast(f"Threshold value set to {threshold_value}. Disparity values below this will be set to 0.")
    else:
        st.error("Please upload a disparity file first.")
        

colId3 = st.columns(5)


resize_disparity = colId3[0].toggle("**Resize Disparity**", value=True)
visualize_disparity = colId3[1].toggle("**Visualize Disparity**", value=False)
gen_cloud = colId3[2].button("Generate Point Cloud")

if visualize_disparity:    
    st.markdown("**Disparity Map Visualization**")
    if disp_path:
        
        disp_temp = disp.copy()
        disp_temp[disp_temp < st.session_state['threshold_value']] = 0
        
        vis = vis_disparity(disp_temp)
        st.image(vis, caption=f"Disparity Map: Threshold value:{st.session_state['threshold_value']}", use_container_width=True)


if resize_disparity:
    if disp_path and calib_file:

        scale_factor = img_size[1] / disp.shape[1]
        
        if scale_factor != (1/calib_parameters["scale_factor_applied"]):
            st.error("The scale factor of the disparity does not match the calibration parameters. Please check your inputs.")
        else:
            if scale_factor != 1:
                disp = resize_2D_data(disp, scale_factor, interpolation_order='cubic')

        
if gen_cloud:
    if disp_path and calib_file:
        try:
            disp[disp < st.session_state['threshold_value']] = 0
            map_x_l, map_y_l, map_x_r, map_y_r, P1, P2, Q = generate_rectify_data(calib_parameters["K1"], calib_parameters["K2"], calib_parameters["R"], calib_parameters["T"], calib_parameters["D1"], calib_parameters["D2"], disp.shape[:2])

            point_cloud = disparity_to_point_cloud(P1, P2, disp)
            cl, ind = point_cloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.5)
            
            point_cloud = point_cloud.select_by_index(ind)
            pcd_outputpath = save_point_cloud(point_cloud, output_base_dir)

            if pcd_outputpath:
                st.success(f"Point cloud saved to: {pcd_outputpath}")
                fig = visualize_point_cloud(point_cloud)
                st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating point cloud: {e}")
    else:
        st.error("Please upload all required files.")