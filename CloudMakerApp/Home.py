import streamlit as st
from streamlit_tile import streamlit_tile

st.set_page_config(
    page_title="Stereo Vision App - Home",
    page_icon="🏠",
    layout="wide"
)

st.title("Cloud Maker 0.0")
st.markdown("This is the **Home page** of the Cloud Maker app. Navigate through the app using the tiles below for **Image Rectification**, **Point Cloud Generation**, **Settings**, and **Disparity Calculation**")

# Navigation tiles
st.subheader("Navigation", divider=True)

col1, col2, col3, col4 = st.columns(4, gap="small")

with col1:

    rectification_clicked = streamlit_tile(
        title="Image Rectification",
        description="Perform image rectification tasks",
        icon="image",
        color_theme="blue",
        key="rectification_tile")

with col2:

    disparity_clicked = streamlit_tile(
        title="Disparity Calculator",
        description="Calculate disparity maps",
        icon="token",
        color_theme="red",
        key="disparity_tile"
    )
    
with col3:
    
    pointcloud_clicked = streamlit_tile(
        title="Point Cloud Generator",
        description="Generate 3D point clouds",
        icon="cloud",
        color_theme="green",
        key="pointcloud_tile"
    )


with col4:
    
    settings_clicked = streamlit_tile(
        title="Settings",
        description="Configure application settings",
        icon="settings",
        color_theme="yellow",
        key="settings_tile"
    )
    

# Handle tile clicks
if rectification_clicked:
    st.info("📷 Navigating to Image Rectification...")
    st.switch_page("pages/01_Image_Rectificator.py")

if disparity_clicked:
    st.info("📏 Navigating to Disparity Calculator...")
    st.switch_page(page="pages/02_Disparity_Calculator.py")

if pointcloud_clicked:
    st.info("☁️ Navigating to Point Cloud Generator...")
    st.switch_page("pages/03_Point_Cloud_Generator.py")

if settings_clicked:
    st.warning("⚙️ Settings page is under construction...")
    # add some symbols of warning or under construction
    st.badge(f":material/engineering:"*32, color="orange")

    # st.switch_page("pages/Settings.py")
