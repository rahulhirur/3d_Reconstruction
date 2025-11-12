import streamlit as st
import json
import plotly.graph_objects as go
import numpy as np

@st.cache_data
def calculate_depth(disparity_map, focal_length, baseline):
    """Calculate depth from disparity map."""
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        depth_map = (focal_length * baseline) / disparity_map
    # Replace inf and NaN values with 0
    
    depth_map[np.isinf(depth_map)] = 0
    depth_map[np.isnan(depth_map)] = 0

    return depth_map.astype(np.float32)

@st.cache_data
def calulcate_disparity(depth_map, focal_length, baseline):
    """Calculate disparity from depth map."""
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        disparity_map = (focal_length * baseline) / depth_map
    # Replace inf and NaN values with 0
    disparity_map[np.isinf(disparity_map)] = 0
    disparity_map[np.isnan(disparity_map)] = 0
    #float32 to save space
    
    return disparity_map.astype(np.float32)

def create_heatmap(data, title):
    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale='Viridis',
        colorbar=dict(title='Value')
    ))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(title=title, xaxis_title='X', yaxis_title='Y')
    return fig

def get_camera_params(calib_data, camera_index=1):
    #get focal length fx from "camera_matrix_1"

    fx1 = calib_data[f"camera_matrix_{camera_index}"][0][0]
    fy1 = calib_data[f"camera_matrix_{camera_index}"][1][1]

    return fx1, fy1

@st.dialog("Visualization Dialog",width="large")
def figViz(fig):
    st.plotly_chart(fig, use_container_width=True)

#create a activate button state in session state
if 'activate_to_visualize' not in st.session_state:
    st.session_state.activate_to_visualize = True

if 'new_d_map' not in st.session_state:
    st.session_state.new_d_map = None

if "fig_title" not in st.session_state:
    st.session_state.fig_title = None

st.set_page_config(page_title="Depth/Disparity Calculator",page_icon=":material/cached:",layout="wide",initial_sidebar_state="collapsed")

st.title("Depth :material/cached: Disparity Calculator")
st.write("This page allows you to convert depth and disparity maps.")

file_type = st.radio(
    "Select the type of file you are uploading:",
    ("Depth", "Disparity"), horizontal=True
)

cols1 = st.columns(2)

uploaded_d_file = cols1[0].file_uploader(f"Upload a {file_type} file", type=["npy"])
uploaded_json_file = cols1[1].file_uploader("Upload a calibration JSON file", type=["json"])

camera_index_list = st.multiselect(
    "Select Camera Index for Calibration Parameters",
    options=[1,2,3,4,5,6] , default=[1])

cols2 = st.columns([1, 1, 1, 5])

if uploaded_d_file is not None and uploaded_json_file is not None:
    try:

        # Load the  map
        d_map = np.load(uploaded_d_file)
        #read the scaled calibration parameters from the JSON file
        calib_data = json.load(uploaded_json_file)

        if file_type == "Depth":
            title = "Depth Map Heatmap"
            to_file_type = "Disparity"
            
        else:
            title = "Disparity Map Heatmap"
            to_file_type = "Depth"

        fig1 = create_heatmap(d_map, title)

        if cols2[0].button(f"View {file_type} Map"):
            figViz(fig1)

        if cols2[1].button(f"Calculate {to_file_type}"):
            
            focal_length = 0
            for camera_index in camera_index_list:
                fx, fy = get_camera_params(calib_data, camera_index=camera_index)
                focal_length += ((fx + fy) / 2)
            focal_length /= len(camera_index_list)  # Average focal length across selected cameras
            
            T = calib_data["Trans_vect"]
            baseline = np.linalg.norm(T)
            
            if file_type == "Depth":
                st.session_state.new_d_map = calulcate_disparity(d_map, focal_length, baseline)
                file_type_new = "Disparity"
                st.session_state.fig_title = "Disparity Map Heatmap"
            else:
                st.session_state.new_d_map = calculate_depth(d_map, focal_length, baseline)
                file_type_new = "Depth"
                st.session_state.fig_title = "Depth Map Heatmap"
            
            st.write("New Map")
            st.write(st.session_state.new_d_map.shape)
            st.write(f"Type: {st.session_state.new_d_map.dtype}")


            st.write(f"Size in MB: {st.session_state.new_d_map.nbytes / (1024 * 1024):.2f} MB")

            #Downlaod the new map as xlsx file
            to_file_name = "/home/hirur/MasterThesis/3d_Reconstruction/Something.csv"
            
            # np.savetxt(to_file_name, st.session_state.new_d_map, delimiter=",")

            #Save as npy file
            to_file_name_npy = to_file_name.replace(".csv", ".npy")
            # np.save(to_file_name_npy, st.session_state.new_d_map)
            
            #Statistics of the new map of depth

            st.write(f"Mean Depth: {np.mean(st.session_state.new_d_map):.2f} meters")
            st.write(f"Max Depth: {np.max(st.session_state.new_d_map):.2f} meters")
            st.write(f"Min Depth: {np.min(st.session_state.new_d_map):.2f} meters")
            #Median
            st.write(f"Median Depth: {np.median(st.session_state.new_d_map):.2f} meters")
            #Standard deviation
            st.write(f"Standard Deviation: {np.std(st.session_state.new_d_map):.2f} meters")

            #agrmax and argmin in 2d array
            st.write(f"Argmax Depth: {np.unravel_index(np.argmax(st.session_state.new_d_map, axis=None), st.session_state.new_d_map.shape)}")

            st.write("Original Map")
            st.write(d_map.shape)
            st.write(f"Type: {d_map.dtype}")
            st.write(f"Size in MB: {d_map.nbytes / (1024 * 1024):.2f} MB")
            

            
            st.session_state.activate_to_visualize = False
            # showcase the depth map as heatmap

        if cols2[2].button(f"View {to_file_type} Map", disabled=st.session_state.activate_to_visualize):
            
            fig2 = create_heatmap(st.session_state.new_d_map, st.session_state.fig_title)
            figViz(fig2)

            # Optionally, display depth statistics
            st.write(f"Mean Depth: {np.mean(st.session_state.new_d_map):.2f} meters")
            st.write(f"Max Depth: {np.max(st.session_state.new_d_map):.2f} meters")
            st.write(f"Min Depth: {np.min(st.session_state.new_d_map):.2f} meters")
        
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON file.")
