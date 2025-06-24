import streamlit as st
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


from disparity_calculator.functions import (initialize_environment, create_output_directory, 
                                            load_configuration, preprocess_images, 
                                            initialize_model, compute_disparity, 
                                            save_disparity)

st.set_page_config(
    page_title="Disparity Calculator",
    page_icon=":material/token:",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title(":material/token: Disparity Calculation")

# You can add some dummy visualization placeholders if you like

# File uploaders for stereo images
col1, col2 = st.columns(2)
left_file = col1.file_uploader("Upload Left Image", type=["png", "jpg", "jpeg"])
right_file = col2.file_uploader("Upload Right Image", type=["png", "jpg", "jpeg"])

# Output directory input
out_dir = st.text_input("Output Directory", value="output/disparity_maps")

# Additional parameters
scale = st.slider("Scale Factor", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
hiera = st.toggle("Hierarchical Disparity Calculation", value=False)

if st.button("Compute Disparity"):
    if left_file and right_file:
        initialize_environment()
        create_output_directory(out_dir)

        args = load_configuration()
        args.left_file = left_file.name
        args.right_file = right_file.name
        args.scale = scale
        args.hiera = hiera

        img0, img1 = preprocess_images(left_file, right_file, args.scale)
        model = initialize_model(args)
        disp = compute_disparity(model, img0, img1, args)

        save_disparity(disp, f"{out_dir}/disp.npy")
        st.success(f"Disparity map saved to {out_dir}/disp.npy")
    else:
        st.error("Please upload both left and right images.")