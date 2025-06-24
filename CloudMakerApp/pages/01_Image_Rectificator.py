import streamlit as st
import plotly.graph_objects as go
import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from image_rectification.functions import CalibrationLoader, resize_image, generate_rectify_data, rectify, save_images, save_scaled_calibration_parameters, create_calibration_data_report, read_image
# Ensure set_page_config is the first Streamlit command
st.set_page_config(page_title="Image Rectificator",page_icon="📷",layout="wide",initial_sidebar_state="collapsed")

# Initialize session state for img1 upload status
if 'Cam_1_uploaded' not in st.session_state:
    st.session_state.Cam_1_uploaded = "Not uploaded"
# Initialize session state for img2 upload status
if 'Cam_2_uploaded' not in st.session_state:
    st.session_state.Cam_2_uploaded = "Not uploaded"
# Initalize session state for calibration file upload status
if 'calib_file_uploaded' not in st.session_state:
    st.session_state.calib_file_uploaded = "Not uploaded"

if 'rectified_img1' not in st.session_state:
    st.session_state.rectified_img1 = None

if 'rectified_img2' not in st.session_state:
    st.session_state.rectified_img2 = None

if "img_size" not in st.session_state:
    st.session_state.img_size = None

if "calibration_param" not in st.session_state:
    st.session_state.calibration_param = None

# Initialize button states in session state
if 'rectify_button_state' not in st.session_state:
    st.session_state.rectify_button_state = False

if 'save_rectified_button_state' not in st.session_state:
    st.session_state.save_rectified_button_state = False

if 'visualize_camera_position_button_state' not in st.session_state:
    st.session_state.visualize_camera_position_button_state = False

@st.cache_data
def streamlit_image_loader(img_path, seesion_id="Cam_1_uploaded"):
    
    if img_path:

        img_x = read_image(img_path)

        if img_x is not None:

            st.session_state[seesion_id] = "Uploaded"
            return img_x
        else:
            st.session_state[seesion_id] = "Failure"
            return None
    else:
        st.session_state[seesion_id] = "Not uploaded"
        return None

@st.cache_data
def streamlit_calibration_loader(calib_file):
    if calib_file:
        
        calib_yaml = CalibrationLoader(calib_file)

        if calib_yaml.is_valid():
            st.session_state.calib_file_uploaded = "Uploaded"
            return calib_yaml
        else:
            st.session_state.calib_file_uploaded = "Failure"
            return None
    else:
        st.session_state.calib_file_uploaded = "Not uploaded"
        return None

def plot_camera_frustum_plotly(R= np.eye(3), t=np.zeros((3, 1)), scale=0.5, fig=None, name="Camera 1", color='blue'):
    """
    Visualizes a 3D camera frustum (field of view) using Plotly.

    Args:
        R (np.array): 3x3 rotation matrix for the camera's orientation in world coordinates.
        t (np.array): 3x1 translation vector for the camera's position in world coordinates.
        scale (float, optional): Scalar to control the size of the camera frustum. Defaults to 0.5.
        fig (go.Figure, optional): An existing Plotly figure to add the frustum to.
                                   If None, a new figure is created. Defaults to None.
        name (str, optional): Name for the camera, used in the legend. Defaults to "Camera 1".
        color (str, optional): Color for the frustum lines and origin marker. Defaults to 'blue'.

    Returns:
        go.Figure: The Plotly figure object with the camera frustum added.
    """
    if fig is None:
        fig = go.Figure()

    # Ensure t is (3,1) for consistent matrix operations
    t = t.reshape(3, 1)

    cam_origin = np.zeros((3, 1))
    frustum_local = np.array([
        [0.5,  0.5, 1],    # Top-right corner
        [-0.5, 0.5, 1],   # Top-left corner
        [-0.5, -0.5, 1],  # Bottom-left corner
        [0.5, -0.5, 1]    # Bottom-right corner
    ]).T * scale  # Transpose to shape (3, 4) and apply scale

    cam_points = np.hstack((cam_origin, frustum_local))  # shape (3, 5)

    world_points = R @ cam_points + t  # shape (3, 5)
    cam_pos = world_points[:, 0]
    
    # Initialize lists to hold all X, Y, Z coordinates for frustum lines
    line_x, line_y, line_z = [], [], []

    #Frustum shape
    for i in range(1, 5):
        line_x.extend([cam_pos[0], world_points[0, i], None])
        line_y.extend([cam_pos[1], world_points[1, i], None])
        line_z.extend([cam_pos[2], world_points[2, i], None])

    for i, j in [(1, 2), (2, 3), (3, 4), (4, 1)]:
        line_x.extend([world_points[0, i], world_points[0, j], None])
        line_y.extend([world_points[1, i], world_points[1, j], None])
        line_z.extend([world_points[2, i], world_points[2, j], None])

    fig.add_trace(go.Scatter3d(
        x=line_x,
        y=line_y,
        z=line_z,
        mode='lines',
        legendgroup=name,
        line=dict(color=color, width=4), # Use the specified color
        name=name,                       # Use the camera name for the legend
        showlegend=True                  # Show legend for this combined trace
    ))

    # Add a marker for the camera position
    fig.add_trace(go.Scatter3d(
        x=[cam_pos[0]],
        y=[cam_pos[1]],
        z=[cam_pos[2]],
        mode='markers',
        legendgroup=name,
        marker=dict(size=4, color=color), # Use the specified color for the marker
        name=name,                        # Same name as frustum lines to merge legend
        showlegend=False                   # Show legend for this marker (will be grouped)
    ))

    # Configure the 3D scene layout and camera perspective
    fig.update_layout(
        # Set the scene camera's 'up' direction and 'eye' position for a specific view
        scene_camera=dict(
            up=dict(x=0, y=1, z=0),
            eye=dict(x=1.25, y=1.25, z=1.25)
        ))
    
    return fig

def create_axis(axis_length=1, fig=None):

    if fig is None:
        fig = go.Figure()
    
    fig.add_trace(go.Scatter3d(
        x=[0, axis_length], y=[0, 0], z=[0, 0],
        mode='lines', line=dict(color='red', width=3), name='X-axis', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, axis_length], z=[0, 0],
        mode='lines', line=dict(color='blue', width=3), name='Y-axis', showlegend=False
    ))
    fig.add_trace(go.Scatter3d(
        x=[0, 0], y=[0, 0], z=[0, axis_length],
        mode='lines', line=dict(color='green', width=3), name='Z-axis', showlegend=False
    ))
    return fig

def update_fig_scene(fig, x_range, y_range, z_range):
    """
    Update the scene layout of the Plotly figure with specified axis ranges.
    
    Args:
        fig (go.Figure): The Plotly figure to update.
        x_range (tuple): Range for the X-axis (min, max).
        y_range (tuple): Range for the Y-axis (min, max).
        z_range (tuple): Range for the Z-axis (min, max).
    """
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=x_range, showgrid=False),
            yaxis=dict(range=y_range, showgrid=False),
            zaxis=dict(range=z_range, showgrid=False),
            aspectmode='auto'
        )
    )

    return fig

def update_fig_width(fig, width):
    """
    Update the width of the Plotly figure.
    
    Args:
        fig (go.Figure): The Plotly figure to update.
        width (int): New width for the figure.
    """
    fig.update_layout(width=width)

    return fig

def update_fig_height(fig, height):
    """
    Update the height of the Plotly figure.
    
    Args:
        fig (go.Figure): The Plotly figure to update.
        height (int): New height for the figure.
    """
    fig.update_layout(height=height)

    return fig

def visualize_camera_position():

    # Create the initial figure with axes
    fig = create_axis(axis_length=0.5)
    # Update the figure title
    fig.update_layout(title="Camera Position Visualization", title_x=0.5)

    fig = update_fig_width(fig, width=800)
    fig = update_fig_height(fig, height=800)

    fig = plot_camera_frustum_plotly(scale=0.25, name="Camera 1", fig=fig, color='blue')

    fig = plot_camera_frustum_plotly(st.session_state.calibration_param["R"], st.session_state.calibration_param["T"], scale=0.25, name="Camera 2", fig=fig, color='red')
    
    fig = update_fig_scene(fig, x_range=[-1.5, 1], y_range=[-1.5, 1], z_range=[-1.5, 1])

    st.plotly_chart(fig, use_container_width=True)

st.title("📷 Image Rectification Page")
st.markdown("""Upload stereo images and a calibration file to perform image rectification. Set output directory, scale factor, and toggle Camera 1 perspective.""")

# File inputs
col1, col2 = st.columns(2)

with col1:
    image1_path = st.file_uploader("**Upload Cam 1 Image**", type=["png", "jpg", "jpeg"])
    
    img_1 = streamlit_image_loader(image1_path, seesion_id="Cam_1_uploaded")
    
    if img_1 is not None:
        if st.session_state.Cam_1_uploaded == "Uploaded":
            st.toast("Cam 1 image uploaded successfully!", icon="✅")
            st.session_state.Cam_1_uploaded = "Not uploaded"
        elif st.session_state.Cam_1_uploaded == "Failure":
            st.error("Error reading Cam 1 image. Please check the file format.")

with col2:
    image2_path = st.file_uploader("**Upload Cam 2 Image**", type=["png", "jpg", "jpeg"])
    
    img_2 = streamlit_image_loader(image2_path, seesion_id="Cam_2_uploaded")

    if img_2 is not None:
        if st.session_state.Cam_2_uploaded == "Uploaded":
            st.session_state.Cam_2_uploaded = "Not uploaded"
            st.toast("Cam 2 image uploaded successfully!", icon="✅")
            
        elif st.session_state.Cam_2_uploaded == "Failure":
            st.error("Error reading Cam 2 image. Please check the file format.")

# Calibration file input
calib_file = st.file_uploader("Upload Calibration JSON File", type="json")

# Load calibration data
calib_yaml = streamlit_calibration_loader(calib_file)

if calib_yaml is not None:
    if st.session_state.calib_file_uploaded == "Uploaded":
        st.session_state.calib_file_uploaded = "Not uploaded"
        st.toast("Calibration file loaded successfully!", icon="✅")
    elif st.session_state.calib_file_uploaded == "Failure":
        st.error("Invalid calibration file. Please upload a valid JSON file.")

# Output directory
output_base_dir = st.text_input("**Output Directory**", value="output/batch_")



# Adjusted layout for scale factor and camera perspective toggle
colId1 = st.columns([9, 1], vertical_alignment="center")

with colId1[0]:
    scale_factor = st.slider("**Scale Factor**", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

with colId1[1]:
    cam1_perspective = st.toggle("**Use Camera 1 Perspective**", value=True)

# Rearranged buttons and toggle horizontally for better layout
colId2 = st.columns([1.5, 1.5, 1.5, 7], vertical_alignment="center")

with colId2[0]:
    if st.button("Rectify Images", key="rectify_button", use_container_width=True):
        st.session_state.rectify_button_state = True

with colId2[1]:
    save_disabled = not st.session_state.rectify_button_state
    if st.button("Save Rectified Images", key="save_rectified_button", use_container_width=True, disabled=save_disabled):
        st.session_state.save_rectified_button_state = True

with colId2[2]:
    visualize_disabled = not st.session_state.rectify_button_state
    if st.button("Visualize Camera Position", key="visualize_camera_position_button", use_container_width=True, disabled=visualize_disabled):
        st.session_state.visualize_camera_position_button_state = True

if st.session_state.rectify_button_state:

    st.session_state.calibration_param = calib_yaml.get_all_calibration(scale_factor, dict_format=True)
    if not image1_path or not image2_path or not calib_file:
        st.error("Please upload all required files.")
    else:
        img_1_resized, img_size_resized = resize_image(img_1, scale_factor)
        img_2_resized, _ = resize_image(img_2, scale_factor)  # Use the size from the first image

        # Update img_size to the new size for subsequent steps like rectification
        st.session_state.img_size = img_size_resized  

        # Get scaled calibration parameters for the new image size
        # K1_scaled, D1_scaled, K2_scaled, D2_scaled, R_scaled, T_scaled = calib_yaml.get_all_calibration(scale_factor, dict_format=True)
        
        if cam1_perspective:
            # Generate rectification maps using the scaled parameters and new image size
            map1x, map1y, map2x, map2y, P1, P2, Q = generate_rectify_data(st.session_state.calibration_param["K1"], st.session_state.calibration_param["K2"], st.session_state.calibration_param["R"], st.session_state.calibration_param["T"], st.session_state.calibration_param["D1"], st.session_state.calibration_param["D2"], st.session_state.img_size)

            # Rectify the resized images
            st.session_state.rectified_img1 = rectify(img_1_resized, map1x, map1y)
            st.session_state.rectified_img2 = rectify(img_2_resized, map2x, map2y)
            
        else:

            R_2_to_1 = st.session_state.calibration_param["R"].T
            T_2_to_1 = -st.session_state.calibration_param["R"].T @ st.session_state.calibration_param["T"]
            # Generate rectify data with right camera as reference
            map_x_new_1, map_y_new_1, map_x_new_2, map_y_new_2, P1_new, P2_new, Q_new = generate_rectify_data(st.session_state.calibration_param["K2"], st.session_state.calibration_param["K1"], R_2_to_1, T_2_to_1, st.session_state.calibration_param["D2"], st.session_state.calibration_param["D1"], st.session_state.img_size)

            st.session_state.rectified_img2 = rectify(img_2_resized, map_x_new_1, map_y_new_1)
            st.session_state.rectified_img1 = rectify(img_1_resized, map_x_new_2, map_y_new_2)

            st.session_state.calibration_param["R"] = R_2_to_1
            st.session_state.calibration_param["T"] = T_2_to_1

        
        st.success("Rectification completed successfully!")

if st.session_state.save_rectified_button_state:
    if not image1_path or not image2_path or not calib_file:
        st.error("Please upload all required files.")
    else:
        
        # Save the rectified images
        save_images(st.session_state.rectified_img1, st.session_state.rectified_img2, output_base_dir)

        # Save scaled calibration parameters
        save_scaled_calibration_parameters(st.session_state.calibration_param["K1"], st.session_state.calibration_param["D1"], st.session_state.calibration_param["K2"], st.session_state.calibration_param["D2"], st.session_state.calibration_param["R"], st.session_state.calibration_param["T"], st.session_state.img_size, scale_factor, output_base_dir)

        # Create and save calibration data report
        
        st.write(f"Current working directory: {os.getcwd()}")
        create_calibration_data_report(st.session_state.calibration_param["K1"], st.session_state.calibration_param["D1"], st.session_state.calibration_param["K2"], st.session_state.calibration_param["D2"], st.session_state.calibration_param["R"], st.session_state.calibration_param["T"], st.session_state.img_size, scale_factor, output_base_dir)

        st.success("Rectified images and calibration data saved successfully!")

if st.session_state.visualize_camera_position_button_state:
    # Check if all required files are uploaded
    if not image1_path or not image2_path or not calib_file:
        st.error("Please upload all required files.")
    else:
        if st.session_state.calibration_param is None:
            st.error("Calibration parameters are not available. Please run rectification first.")
        else:
            visualize_camera_position()
