import streamlit as st
import pydicom
import numpy as np
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas

# Configure page layout
st.set_page_config(page_title="RaySafe Phantom IQ Analysis", layout="wide")

# Initialize session state variables (equivalent to fig.UserData)
if 'image' not in st.session_state:
    st.session_state.image = None
if 'image0' not in st.session_state:
    st.session_state.image0 = None
if 'info' not in st.session_state:
    st.session_state.info = {}
if 'log' not in st.session_state:
    st.session_state.log = ["Please load an image to begin."]

def append_log(message):
    st.session_state.log.append(message)

# Sidebar for Controls (Replacing standard UI buttons)
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Load Image (DICOM, PNG, JPG)", type=['dcm', 'png', 'jpg', 'tif', 'bmp'])
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Flip IMG UD"):
            if st.session_state.image is not None:
                st.session_state.image = np.flipud(st.session_state.image)
                st.session_state.image0 = np.flipud(st.session_state.image0)
    with col2:
        if st.button("Rotate IMG"):
            if st.session_state.image is not None:
                st.session_state.image = np.rot90(st.session_state.image)
                st.session_state.image0 = np.rot90(st.session_state.image0)
                
    st.markdown("---")
    if st.button("Compute MTF"):
        st.session_state.action = "compute_mtf"
    if st.button("Compute CNR"):
        st.session_state.action = "compute_cnr"

# Main processing logic
if uploaded_file is not None:
    # Only load if it's a new file
    if st.session_state.image is None:
        try:
            if uploaded_file.name.lower().endswith('.dcm'):
                # Read DICOM using pydicom
                ds = pydicom.dcmread(uploaded_file)
                st.session_state.image0 = ds.pixel_array.astype(float)
                
                # Normalize image (mat2gray equivalent)
                img_min = np.min(st.session_state.image0)
                img_max = np.max(st.session_state.image0)
                st.session_state.image = (st.session_state.image0 - img_min) / (img_max - img_min)
                
                # Extract metadata based on your MATLAB logic
                px = getattr(ds, 'PixelSpacing', [0.15, 0.15])[0]
                kv = getattr(ds, 'KVP', 'N/A')
                
                # Store info
                st.session_state.info = {'PixelSpacing': px, 'KVP': kv}
                append_log(f"Image loaded, px={px:.3f} mm/pix")
                append_log(f"kV = {kv}")
                
        except Exception as e:
            st.error(f"Load error: {e}")

# Layout for Main Screen
col_img, col_results = st.columns([1.5, 1])

with col_img:
    st.subheader("Input Image (Draw ROIs here)")
    if st.session_state.image is not None:
        # Scale image for display in canvas
        display_img = (st.session_state.image * 255).astype(np.uint8)
        
        # Create a canvas for drawing ROIs
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=2,
            stroke_color="red",
            background_image=plt.imread(uploaded_file) if not uploaded_file.name.endswith('.dcm') else None, # Placeholder for non-dicom
            update_streamlit=True,
            height=512,
            width=512,
            drawing_mode="rect",
            key="canvas",
        )

with col_results:
    st.subheader("Analysis Log")
    st.text_area("Log Output", value="\n".join(st.session_state.log), height=200, disabled=True)
    
    st.subheader("MTF Plot")
    # Placeholder for the matplotlib axis (axMTF)
    fig, ax = plt.subplots()
    ax.set_title("Post-Sampled MTF")
    ax.set_xlabel("Frequency (lp/mm)")
    ax.set_ylabel("MTF")
    st.pyplot(fig)