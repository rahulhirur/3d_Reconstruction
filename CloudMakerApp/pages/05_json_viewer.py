import streamlit as st
import json
import plotly.graph_objects as go

st.title("JSON Viewer")

# File uploader for JSON file
uploaded_file = st.file_uploader("Upload a JSON file", type=["json"])

if uploaded_file is not None:
    try:
        # Load and parse the JSON file
        json_data = json.load(uploaded_file)
        
        # Display the JSON content
        st.subheader("JSON Content")
        st.json(json_data)
    except json.JSONDecodeError:
        st.error("Invalid JSON file. Please upload a valid JSON file.")
