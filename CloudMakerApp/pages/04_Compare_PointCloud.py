import streamlit as st
import numpy as np

from plyfile import PlyData, PlyElement

from scipy.spatial import cKDTree # For efficient nearest neighbor search (if needed separately)

from pcd_comparison.functions import load_and_preprocess_pcd, generate_transformation_init, icp_align_and_compare, save_comparison_results

#page layout wide

st.set_page_config(
    page_title="PLY File Shape Comparison",
    page_icon=":material/compare:",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.title("PLY File Shape Comparison Tool")

st.markdown("""
Upload two PLY files to compare their shapes. The tool will:
1.  Load the point clouds.
2.  Optionally downsample them for faster processing.
3.  Attempt to align them using Iterative Closest Point (ICP) registration.
4.  Calculate the average geometric distance between the aligned shapes.

**Important:** For meaningful shape comparison, the two PLY files should represent the *same object or scene*. ICP works best with a good initial alignment.
""")

col1 = st.columns(2)

# --- File Uploaders ---
col1[0].subheader("Upload PLY File 1 (Source)")
uploaded_file1 = col1[0].file_uploader("Choose the first PLY file", type="ply", key="file1")

col1[1].subheader("Upload PLY File 2 (Target)")
uploaded_file2 = col1[1].file_uploader("Choose the second PLY file", type="ply", key="file2")

output_base_dir = st.text_input("**Output Directory**")

# --- Parameters for comparison ---
st.sidebar.header("Comparison Settings")

icp_threshold = st.sidebar.slider(
    "ICP Correspondence Distance (m)",
    min_value=0.001, max_value=1.0, value=0.02, step=0.001,
    help="Maximum distance between corresponding points for ICP. Adjust based on scale of your objects."
)

#Create a radio to select a method to initialize trans_init
st.sidebar.subheader("Transformation Initialization Method")

trans_init_method = st.sidebar.radio(
    "Select Method",
    options=["Identity", "Random"],
    help="Choose how to initialize the transformation matrix for ICP.")

if uploaded_file1 is not None and uploaded_file2 is not None:
    st.header("Shape Comparison Results")

    # Load and preprocess both point clouds
    with st.spinner("Loading and preprocessing point clouds..."):
        source_pcd = load_and_preprocess_pcd(uploaded_file1)
        target_pcd = load_and_preprocess_pcd(uploaded_file2)

        # target_pcd = add_noise_to_pcd(target_pcd, std_dev=0.01)

        #mean of the target point cloud
        target_mean = np.mean(np.asarray(target_pcd.points), axis=0)
        #center the target point cloud
        st.write(f"Target Point Cloud Mean: {target_mean.round(3)}")

        #center the source point cloud
        source_mean = np.mean(np.asarray(source_pcd.points), axis=0)
        st.write(f"Source Point Cloud Mean: {source_mean.round(3)}")

        #Add noise to the target point cloud

    if source_pcd is not None and target_pcd is not None:
        st.subheader("1. Point Cloud Information")
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**{uploaded_file1.name} (Source)**")
            st.write(f"Points: {len(source_pcd.points)}")
            st.write(f"Min Bounds: {np.asarray(source_pcd.get_min_bound()).round(3)}")
            st.write(f"Max Bounds: {np.asarray(source_pcd.get_max_bound()).round(3)}")
        
        with col2:
            st.write(f"**{uploaded_file2.name} (Target)**")
            st.write(f"Points: {len(target_pcd.points)}")
            st.write(f"Min Bounds: {np.asarray(target_pcd.get_min_bound()).round(3)}")
            st.write(f"Max Bounds: {np.asarray(target_pcd.get_max_bound()).round(3)}")

        st.subheader("2. ICP Registration (Alignment)")
        st.info("ICP will try to align the 'Source' to the 'Target'. A good initial guess for alignment can significantly improve results.")
        
        trans_init_matrix = generate_transformation_init(trans_init_method)

        if st.button("Calculate ICP RMSE Error"):
            with st.spinner("Running ICP... This might take a moment for large clouds."):
                result_transformation, rmse, chamfer, hausdorff = icp_align_and_compare(source_pcd, target_pcd, threshold=icp_threshold, trans_init= trans_init_matrix)

            st.success("ICP Complete!")
            
            st.subheader("Results")

            # st.success(f"Fitness (overlap): {result.fitness}")
            # st.success(f"Inlier RMSE (Root Mean Square Error): {result.inlier_rmse}")
            st.write("Transformation Matrix:")
            st.write(result_transformation)

            st.write(f"rmse, chamfer, hausdorff {rmse, chamfer, hausdorff}")

            # Save the comparison results
            if output_base_dir:
                save_comparison_results(output_base_dir, result_transformation, rmse, chamfer, hausdorff)
                st.success(f"Comparison results saved to {output_base_dir}")
            
            # #Visualize the aligned point cloud
            # transformed_source_pcd = source_pcd.transform(result_transformation)
            
            # # Extract point coordinates for plotting
            # source_points = np.asarray(transformed_source_pcd.points)
            # target_points = np.asarray(target_pcd.points)
            
            # # Create a Plotly Scatter3d trace for the transformed source point cloud
            # source_trace = go.Scatter3d(
            #     x=source_points[:, 0],
            #     y=source_points[:, 1],
            #     z=source_points[:, 2],
            #     mode='markers',
            #     marker=dict(size=2, color='red'),
            #     name='Source (Transformed)'
            # )
            
            # # Create a Plotly Scatter3d trace for the target point cloud
            # target_trace = go.Scatter3d(
            #     x=target_points[:, 0],
            #     y=target_points[:, 1],
            #     z=target_points[:, 2],
            #     mode='markers',
            #     marker=dict(size=2, color='blue'),
            #     name='Target'
            # )

            # # Create a figure with both traces
            # fig = go.Figure(data=[source_trace, target_trace])
            # fig.update_layout(title='Aligned Point Clouds')

            # # # Display the interactive Plotly chart in Streamlit
            # # st.plotly_chart(fig)


            # # Apply the transformation to the source point cloud
            
else:
    st.info("Please upload two PLY files to start the shape comparison.")
