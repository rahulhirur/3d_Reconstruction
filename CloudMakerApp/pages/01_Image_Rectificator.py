import streamlit as st
import plotly.graph_objects as go
import os
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from image_rectification.functions import CalibrationLoader, resize_image, generate_rectify_data, rectify, save_images, save_scaled_calibration_parameters, create_calibration_data_report, read_image, suggest_next_folder_name, transpose_image_size, save_rectification_artifacts
import json

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

if "img1_size" not in st.session_state:
    st.session_state.img1_size = None

if "calibration_param" not in st.session_state:
    st.session_state.calibration_param = None

if "P1" not in st.session_state:
    st.session_state.P1 = None

if "P2" not in st.session_state:
    st.session_state.P2 = None

# Initialize button states in session state
if 'rectify_button_state' not in st.session_state:
    st.session_state.rectify_button_state = False

if 'save_rectified_button_state' not in st.session_state:
    st.session_state.save_rectified_button_state = False

if 'save_rectified_button_activate_state' not in st.session_state:
    st.session_state.save_rectified_button_activate_state = False

if 'visualize_camera_position_button_state' not in st.session_state:
    st.session_state.visualize_camera_position_button_state = False

if 'visualize_camera_position_button_activate_state' not in st.session_state:
    st.session_state.visualize_camera_position_button_activate_state = False

if 'visualize_rectification_images_button_state' not in st.session_state:
    st.session_state.visualize_rectification_images_button_state = False

if 'visualize_rectification_images_button_activate_state' not in st.session_state:
    st.session_state.visualize_rectification_images_button_activate_state = False

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

def get_images_from_folder(folder_path):
    """
    Retrieve a list of PNG images from the specified folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        list: List of PNG image file paths.
    """
    if folder_path:
        try:

            file_names = [f for f in os.listdir(folder_path) if f.lower().endswith(".png")]
            #sort the file names
            file_names.sort()  # Sort the file names alphabetically
            
            folder_file_paths = [os.path.join(folder_path, f) for f in file_names]
            return folder_file_paths, file_names
        except Exception as e:
            st.error(f"Error reading folder: {e}")
            return [],[]
    return [],[]

def rectification_quality_analysis(img_size, evaluation_data):

    st.info(f"Camera 1: **{evaluation_data['roi1_percentage']:.2f}%** | Camera 2: **{evaluation_data['roi2_percentage']:.2f}%** of the image area is used in rectification.")
        
    
    img_width, img_height = img_size
    
    pp1_x, pp1_y = evaluation_data["principal_point1"]
    pp2_x, pp2_y = evaluation_data["principal_point2"]

    pp1_status = "inside" if 0 <= pp1_x < img_width and 0 <= pp1_y < img_height else "outside"
    pp2_status = "inside" if 0 <= pp2_x < img_width and 0 <= pp2_y < img_height else "outside"

    if pp1_status == "inside":
        st.info(f"Cam 1 Principal Point: ({pp1_x:.2f}, {pp1_y:.2f}) is {pp1_status}. Image Size: {img_width}x{img_height}.")
    else:
        st.warning(f"Cam 1 Principal Point: ({pp1_x:.2f}, {pp1_y:.2f}) is {pp1_status}. Image Size: {img_width}x{img_height}.")

    if pp2_status == "inside":
        st.info(f"Cam 2 Principal Point: ({pp2_x:.2f}, {pp2_y:.2f}) is {pp2_status}. Image Size: {img_width}x{img_height}.")
    else:
        st.warning(f"Cam 2 Principal Point: ({pp2_x:.2f}, {pp2_y:.2f}) is {pp2_status}. Image Size: {img_width}x{img_height}.")
    
    # visualize the principal points on the rectangle of size img_size
    figx = go.Figure()
    
    figx.add_shape(type="rect", x0=0, y0=0, x1=img_width, y1=img_height, line=dict(color="purple", width=5), fillcolor="rgba(255,255,255,255)", name="Image Size")
    
    figx.add_trace(go.Scatter(x=[pp1_x], y=[pp1_y],
                            mode='markers+text',
                            marker=dict(color='blue', size=10),
                            text=[f"PP1 ({pp1_x:.2f}, {pp1_y:.2f})"],
                            textposition="top center",
                            name="Principal Point 1"))
    figx.add_trace(go.Scatter(x=[pp2_x], y=[pp2_y],
                            mode='markers+text',
                            marker=dict(color='red', size=10),
                            text=[f"PP2 ({pp2_x:.2f}, {pp2_y:.2f})"],
                            textposition="top center",
                            name="Principal Point 2"))
    
    # Set the layout for the figure
    figx.update_layout(title="Principal Points on Rectified Image Size", xaxis_title="Image Width", yaxis_title="Image Height")

    figx.update_xaxes(range=[min(pp1_x, pp2_x, 0) - 100, max(pp1_x, pp2_x, img_width) + 100], showgrid=False)
    figx.update_yaxes(range=[min(pp1_y, pp2_y, 0) - 100, max(pp1_y, pp2_y, img_height) + 100], showgrid=False, zeroline=False)

    st.plotly_chart(figx, use_container_width=True)

st.title("📷 Image Rectification Page")
st.markdown("""Upload stereo images and a calibration file to perform image rectification. Set output directory, scale factor, and toggle Camera 1 perspective.""")

# File inputs

folder_mode = st.radio("Select Input Mode", ("Folder Selection", "File Upload"), index=0, horizontal=True)
col1, col2 = st.columns(2)

# File uploaders for Cam 1 and Cam 2
with col1:

    if folder_mode == "Folder Selection":

        folder_path_cam1 = st.text_input("**Select Folder Containing Images for Cam 1**", value="")
        available_images_cam1, available_images_names_cam1 = get_images_from_folder(folder_path_cam1)

        if len(available_images_cam1) > 0:
            image1_path = st.selectbox("**Select Cam 1 Image**", options=available_images_cam1, format_func=lambda x: available_images_names_cam1[available_images_cam1.index(x)])
            # image1_path = st.selectbox("**Select Cam 1 Image**", options=available_images_cam1)
            img_1 = streamlit_image_loader(image1_path, seesion_id="Cam_1_uploaded")
        else:
            image1_path = None

    else:
        image1_path = st.file_uploader("**Upload Cam 1 Image**", type=["png", "jpg", "jpeg"])
        img_1 = streamlit_image_loader(image1_path, seesion_id="Cam_1_uploaded")
        
    #check if img_1 variable exists or not

    if "img_1" in locals():
        if img_1 is not None:
            if st.session_state.Cam_1_uploaded == "Uploaded":

                st.toast("Cam 1 image uploaded successfully!", icon="✅")
                st.session_state.img1_size = [img_1.shape[1],img_1.shape[0]]
                st.session_state.Cam_1_uploaded = "Not uploaded"

            elif st.session_state.Cam_1_uploaded == "Failure":
                st.error("Error reading Cam 1 image. Please check the file format.")

with col2:

    if folder_mode == "Folder Selection":

        folder_path_cam2 = st.text_input("**Select Folder Containing Images for Cam 2**", value="")
        available_images_cam2, available_images_names_cam2 = get_images_from_folder(folder_path_cam2)

        if len(available_images_cam2) > 0:
            image2_path = st.selectbox("**Select Cam 2 Image**", options=available_images_cam2, format_func=lambda x: available_images_names_cam2[available_images_cam2.index(x)])
            # image2_path = st.selectbox("**Select Cam 2 Image**", options=available_images_cam2)
            img_2 = streamlit_image_loader(image2_path, seesion_id="Cam_2_uploaded")
        else:
            image2_path = None

    else:
            
        image2_path = st.file_uploader("**Upload Cam 2 Image**", type=["png", "jpg", "jpeg"])
        img_2 = streamlit_image_loader(image2_path, seesion_id="Cam_2_uploaded")
    
    if "img_2" in locals():
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
    # Retrieve the number of cameras and reference camera index
    num_cameras = calib_yaml.get_num_cameras()
    ref_camera_index = calib_yaml.get_reference_camera_index()

    # Display the number of cameras and reference camera index in one line
    st.info(f"*Number of cameras:* **{num_cameras}** | *Reference camera index:* **{ref_camera_index}**")

    # Handle calibration file upload status
    if st.session_state.calib_file_uploaded == "Uploaded":
        st.session_state.calib_file_uploaded = "Not uploaded"
        st.toast("Calibration file loaded successfully!", icon="✅")
    elif st.session_state.calib_file_uploaded == "Failure":
        st.error("Invalid calibration file. Please upload a valid JSON file.")

    # Allow selection of camera indices regardless of the number of cameras
    col_cam1, col_cam2 = st.columns(2)

    # Dropdown for selecting Camera 1 index
    with col_cam1:
        cam1_index = st.selectbox("Select Camera 1 Index", options=list(range(num_cameras)), index=ref_camera_index)

    # Dropdown for selecting Camera 2 index
    with col_cam2:
        cam2_index = st.selectbox("Select Camera 2 Index", options=list(range(num_cameras)), index=(ref_camera_index + 1) % num_cameras)

    # Update calibration parameters based on selected camera indices
    # st.session_state.calibration_param = calib_yaml.get_calibration_for_cameras_by_index(cam1_index, cam2_index, scale_factor=1.0, dict_format=True)

# Output directory
output_base_dir = st.text_input("**Output Directory**", value=suggest_next_folder_name("output", "batch_"))

# Adjusted layout for scale factor and camera perspective toggle
colId1 = st.columns([4,1,1,2,1], vertical_alignment="top")

with colId1[0]:
    scale_factor = st.slider("**Scale Factor**", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

with colId1[1]:
    # Create selectbox for predefined width options

    if st.session_state.img1_size is not None and img_1 is not None:

        default_widthx = st.session_state.img1_size[0] * 0.25
        
        predefined_widths = [
            default_widthx * 1.00,
            default_widthx * 1.25,
            default_widthx * 1.5,
            default_widthx * 1.75,
            default_widthx * 2.0,
            "Custom..."
        ]

        width_selection = st.selectbox(
            "**Select New Width**",
            options=predefined_widths,
            format_func=lambda x: f"{int(x)}" if x != "Custom..." else x)

        # Handle custom width input
        if width_selection == "Custom...":
            new_rectification_width = int(st.text_input("Enter custom width...", value=str(int(default_widthx))))
        else:
            new_rectification_width = int(width_selection)
    else:
        new_rectification_width = int(st.text_input("**New Width**", value="1368"))
    
    # if st.session_state.img1_size is not None:
    #     default_width = st.session_state.img1_size[0] *0.25
    #     # st.write(st.session_state.img1_size)
    # else:
    #     default_width = 800
    
    # new_rectification_width = int(st.text_input("**New Width**", value=str(default_width)))
    # default_width = img1_size[0] if "img1_size" in locals() and img1_size else 800
    # new_rectification_width = int(st.text_input("**New Width**", value=str(default_width)))

with colId1[2]:

    if st.session_state.img1_size is not None and img_1 is not None:

        default_heightx = st.session_state.img1_size[1] * 0.25

        predefined_heights = [
            default_heightx * 1.00,
            default_heightx * 1.25,
            default_heightx * 1.5,
            default_heightx * 1.75,
            default_heightx * 2.0,
            "Custom..."
        ]

        height_selection = st.selectbox(
            "**Select New Height**",
            options=predefined_heights,
            format_func=lambda x: f"{int(x)}" if x != "Custom..." else x)

        # Handle custom height input

        if height_selection == "Custom...":
            new_rectification_height = int(st.text_input("Enter custom height...", value=str(int(default_heightx))))

        else:
            new_rectification_height = int(height_selection)

    else:
        new_rectification_height = int(st.text_input("**New height**", value="912"))

with colId1[3]:

    stereoRectify_flags_labels = ["cv2.CALIB_ZERO_TANGENT_DIST", "cv2.CALIB_SAME_FOCAL_LENGTH", "cv2.CALIB_ZERO_DISPARITY"]
    stereoRectify_flags = [8, 512,1024]

    selected_stereoRectify_flag = st.selectbox(
        "**Rectification Flags**",
        options=stereoRectify_flags,
        index=0,
        format_func=lambda x: stereoRectify_flags_labels[stereoRectify_flags.index(x)]
    )
    # st.write(selected_stereoRectify_flag)

with colId1[4]:
    cam1_perspective = st.toggle("**Use Camera 1 Perspective**", value=True)

colId2 = st.columns([1.5, 1.5, 1.5, 2, 6], vertical_alignment="center")

with colId2[0]:
    rectify_disabled = not image1_path or not image2_path or not calib_file
    if st.button("Rectify Images", key="rectify_button", use_container_width=True, disabled= rectify_disabled):
        st.session_state.rectify_button_state = True
        st.session_state.visualize_camera_position_button_activate_state = True
        st.session_state.visualize_rectification_images_button_activate_state = True
        st.session_state.save_rectified_button_activate_state = True

with colId2[1]:
    
    if st.button("Save Rectified Images", key="save_rectified_button", use_container_width=True, disabled=not st.session_state.save_rectified_button_activate_state):
        st.session_state.save_rectified_button_state = True
        
with colId2[2]:

    visualize_disabled = not st.session_state.visualize_camera_position_button_activate_state

    if st.button("Visualize Camera Position", key="visualize_camera_position_button", use_container_width=True, disabled=visualize_disabled):
        st.session_state.visualize_camera_position_button_state = True

with colId2[3]:
    visualize_rectification_images = not st.session_state.visualize_rectification_images_button_activate_state

    if st.button("Visualize Rectification Images", key="visualize_rectification_images_button", use_container_width=True, disabled=visualize_rectification_images):
        st.session_state.visualize_rectification_images_button_state = True

if st.session_state.rectify_button_state:
    
    if not image1_path or not image2_path or not calib_file:
        st.error("Please upload all required files.")

    else:
        
        st.session_state.calibration_param = calib_yaml.get_all_calibration(cam1_index, cam2_index, scale_factor, dict_format=True)
        
        img_1_resized, img_size_resized = resize_image(img_1, scale_factor)
        img_2_resized, _ = resize_image(img_2, scale_factor)  # Use the size from the first image
        
        # Update img_size to the new size for subsequent steps like rectification
        st.session_state.img_size = transpose_image_size(img_size_resized)  
        
        if cam1_perspective:
            # Generate rectification maps using the scaled parameters and new image size
            new_rectification_img_size = [new_rectification_width, new_rectification_height]
            map1x, map1y, map2x, map2y, P1, P2, Q, evaluation_data = generate_rectify_data(st.session_state.calibration_param["K1"], st.session_state.calibration_param["K2"], st.session_state.calibration_param["R"], st.session_state.calibration_param["T"], st.session_state.calibration_param["D1"], st.session_state.calibration_param["D2"], st.session_state.img_size, selected_stereoRectify_flag, new_rectification_img_size)
            
            st.session_state.P1 = P1
            st.session_state.P2 = P2
            
            st.session_state.rectified_img1 = rectify(img_1_resized, map1x, map1y)
            st.session_state.rectified_img2 = rectify(img_2_resized, map2x, map2y)
             
        else:

            R_2_to_1 = st.session_state.calibration_param["R"].T
            T_2_to_1 = -st.session_state.calibration_param["R"].T @ st.session_state.calibration_param["T"]
            
            # Generate rectify data with right camera as reference
            new_rectification_img_size = [new_rectification_width, new_rectification_height]
            map_x_new_1, map_y_new_1, map_x_new_2, map_y_new_2, P1_new, P2_new, Q_new, evaluation_data = generate_rectify_data(st.session_state.calibration_param["K2"], st.session_state.calibration_param["K1"], R_2_to_1, T_2_to_1, st.session_state.calibration_param["D2"], st.session_state.calibration_param["D1"], st.session_state.img_size, selected_stereoRectify_flag, new_rectification_img_size)

            st.session_state.P1 = P1_new
            st.session_state.P2 = P2_new

            st.session_state.rectified_img2 = rectify(img_2_resized, map_x_new_1, map_y_new_1)
            st.session_state.rectified_img1 = rectify(img_1_resized, map_x_new_2, map_y_new_2)
            
            st.session_state.calibration_param["R"] = R_2_to_1
            st.session_state.calibration_param["T"] = T_2_to_1
        
        # Check if the principal points lie inside the rectified images
        rectification_quality_analysis(st.session_state.img_size, evaluation_data)

        st.session_state.rectify_button_state = False  # Reset the button state after rectification        
        st.success("Rectification completed successfully!")

if st.session_state.save_rectified_button_state:

    if not image1_path or not image2_path or not calib_file:
        st.error("Please upload all required files.")
    else:
        
        # Save the rectified images
        save_images(st.session_state.rectified_img1, st.session_state.rectified_img2, output_base_dir)

        save_rectification_artifacts(st.session_state.P1, st.session_state.P2, output_base_dir)

        # Save scaled calibration parameters
        save_scaled_calibration_parameters(st.session_state.calibration_param["K1"], st.session_state.calibration_param["D1"], st.session_state.calibration_param["K2"], st.session_state.calibration_param["D2"], st.session_state.calibration_param["R"], st.session_state.calibration_param["T"], st.session_state.img_size, scale_factor, output_base_dir)
        # Create and save calibration data report
        
        st.write(f"Current working directory: {os.getcwd()}")
        create_calibration_data_report(st.session_state.calibration_param["K1"], st.session_state.calibration_param["D1"], st.session_state.calibration_param["K2"], st.session_state.calibration_param["D2"], st.session_state.calibration_param["R"], st.session_state.calibration_param["T"], st.session_state.img_size, scale_factor, output_base_dir)

        st.success("Rectified images and calibration data saved successfully!")
        st.session_state.save_rectified_button_state = False  # Reset the button state after saving
        st.session_state.save_rectified_button_activate_state = False
        st.rerun()

if st.session_state.visualize_camera_position_button_state:
    # Check if all required files are uploaded

    if not image1_path or not image2_path or not calib_file:
        st.error("Please upload all required files.")
    else:
        if st.session_state.calibration_param is None:
            st.error("Calibration parameters are not available. Please run rectification first.")
        else:
            visualize_camera_position()
            st.session_state.visualize_camera_position_button_state = False

if st.session_state.visualize_rectification_images_button_state:
    # Check if all required files are uploaded

    if not image1_path or not image2_path or not calib_file:
        st.error("Please upload all required files.")
    else:
        if st.session_state.calibration_param is None:
            st.error("Calibration parameters are not available. Please run rectification first.")
        else:
            if st.session_state.rectified_img1 is not None and st.session_state.rectified_img2 is not None:
                col_img1, col_img2 = st.columns(2)
                with col_img1:
                    st.image(st.session_state.rectified_img1, caption="Rectified Image 1", use_container_width=True)
                with col_img2:
                    st.image(st.session_state.rectified_img2, caption="Rectified Image 2", use_container_width=True)
            else:
                st.error("Rectified images are not available. Please run rectification first.")
            st.session_state.visualize_rectification_images_button_state = False
