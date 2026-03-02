import streamlit as st
import numpy as np
import pydicom
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_drawable_canvas import st_canvas
import io

# Import your custom calculation engines
# (Ensure mtf_module.py and cnr_module.py are in the same directory)
from mtf_module import fit_two_gauss_unsharp_mask_mtf0
from cnr_module import wedge_segmentation, calculate_cnr

# --- Configuration & State Management ---
st.set_page_config(page_title="RaySafe Phantom IQ Analysis", layout="wide")

if 'image' not in st.session_state:
    st.session_state.image = None
if 'image0' not in st.session_state:
    st.session_state.image0 = None
if 'info' not in st.session_state:
    st.session_state.info = {}
if 'log' not in st.session_state:
    st.session_state.log = ["System initialized. Please load an image to begin."]
if 'results' not in st.session_state:
    st.session_state.results = {}

def append_log(message):
    st.session_state.log.append(message)

# --- Helper Functions ---
def parse_dicom(file_obj):
    """Replicates the MATLAB dicominfo fallback logic for highest accuracy."""
    ds = pydicom.dcmread(file_obj)
    info = {}
    
    # Pixel Spacing
    if 'PixelSpacing' in ds:
        info['px'] = float(ds.PixelSpacing[0])
    elif 'ImagerPixelSpacing' in ds:
        info['px'] = float(ds.ImagerPixelSpacing[0])
    elif 'SpatialResolution' in ds:
        info['px'] = float(ds.SpatialResolution[0])
    else:
        info['px'] = 0.15
        
    # Focal Spot
    if 'FocalSpots' in ds:
        info['FocalSpot'] = float(ds.FocalSpots[0] if isinstance(ds.FocalSpots, list) else ds.FocalSpots)
    elif 'XRayFocalSpot' in ds:
        info['FocalSpot'] = float(ds.XRayFocalSpot)
    else:
        info['FocalSpot'] = None

    # kV and mAs
    info['kV'] = float(ds.KVP) if 'KVP' in ds else None
    
    if 'ExposureInuAs' in ds:
        info['mAs'] = float(ds.ExposureInuAs) / 10**3
    elif 'ExposureTimeInuS' in ds and 'XRayTubeCurrentInuA' in ds:
        info['mAs'] = (float(ds.ExposureTimeInuS) / 10**6) * (float(ds.XRayTubeCurrentInuA) / 10**3)
    elif 'ExposureTime' in ds and 'XRayTubeCurrent' in ds:
        info['mAs'] = float(ds.ExposureTime) * float(ds.XRayTubeCurrent)
    elif 'Exposure' in ds:
        info['mAs'] = float(ds.Exposure)
    else:
        info['mAs'] = None
        
    # Metadata for logs/export
    info['PatientID'] = str(getattr(ds, 'PatientID', 'Unknown_ID'))
    info['StudyDate'] = str(getattr(ds, 'StudyDate', 'Unknown_Date'))
    info['Manufacturer'] = str(getattr(ds, 'Manufacturer', 'Unknown_Mfg'))
    
    # Image arrays
    img0 = ds.pixel_array.astype(float)
    img_disp = (img0 - np.min(img0)) / (np.max(img0) - np.min(img0)) # mat2gray equivalent
    
    return img0, img_disp, info

# --- Sidebar Controls ---
with st.sidebar:
    st.header("1. Data Input")
    uploaded_file = st.file_uploader("Load DICOM Image", type=['dcm'])
    
    if uploaded_file is not None and st.session_state.image is None:
        try:
            img0, img_disp, info = parse_dicom(uploaded_file)
            st.session_state.image0 = img0
            st.session_state.image = img_disp
            st.session_state.info = info
            
            append_log(f"Image loaded successfully. px={info['px']:.3f} mm/pix")
            if info['kV']: append_log(f"kV = {info['kV']:.1f}")
            if info['mAs']: append_log(f"mAs = {info['mAs']:.2f}")
            if info['FocalSpot']: append_log(f"Nominal Focal Spot = {info['FocalSpot']:.1f} mm")
            append_log("Please proceed to MTF or CNR calculation.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load DICOM: {e}")

    st.header("2. Image Adjustments")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Flip IMG UD", use_container_width=True) and st.session_state.image is not None:
            st.session_state.image = np.flipud(st.session_state.image)
            st.session_state.image0 = np.flipud(st.session_state.image0)
            st.rerun()
    with col2:
        if st.button("Rotate IMG", use_container_width=True) and st.session_state.image is not None:
            st.session_state.image = np.rot90(st.session_state.image)
            st.session_state.image0 = np.rot90(st.session_state.image0)
            st.rerun()

    st.header("3. Analysis")
    compute_mtf = st.button("Compute MTF", use_container_width=True, type="primary")
    compute_cnr = st.button("Compute CNR", use_container_width=True, type="primary")

# --- Main Dashboard Layout ---
col_img, col_results = st.columns([1.2, 1])

# Left Column: Image and Canvas
with col_img:
    st.subheader("Input Image & ROI Selection")
    st.markdown("Draw bounding boxes over the resolution patterns. *Group 1 (0.6-1.6 lp/mm) and Group 2 (1.8-5.0 lp/mm)*.")
    
    if st.session_state.image is not None:
        # Scale to uint8 for the canvas background
        bg_image = (st.session_state.image * 255).astype(np.uint8)
        h, w = bg_image.shape
        
        # Display the canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.2)",
            stroke_width=2,
            stroke_color="red",
            background_image=fromarray(bg_image) if 'fromarray' in globals() else None, # Will utilize PIL internally in typical deployment
            update_streamlit=True,
            height=h // 2, # Scale down visually to fit screen
            width=w // 2,
            drawing_mode="rect",
            key="roi_canvas",
        )
        # Note: If PIL is needed for the background image, ensure `from PIL.Image import fromarray` is added.

# Right Column: Logs, Results, and Exports
with col_results:
    st.subheader("Analysis Log")
    st.text_area("Event Log", value="\n".join(st.session_state.log), height=150, disabled=True)
    
    # --- Execution Logic: MTF ---
    if compute_mtf:
        if canvas_result.json_data is None or len(canvas_result.json_data["objects"]) == 0:
            st.warning("Please draw ROIs on the image first.")
        else:
            append_log("MTF calculation started...")
            # Extract ROI coordinates from canvas JSON
            rois = []
            for obj in canvas_result.json_data["objects"]:
                if obj["type"] == "rect":
                    # Canvas is scaled by 0.5, multiply by 2 to get original array coordinates
                    left = int(obj["left"] * 2)
                    top = int(obj["top"] * 2)
                    width = int(obj["width"] * 2)
                    height = int(obj["height"] * 2)
                    rois.append((left, top, width, height))
            
            append_log(f"Extracted {len(rois)} ROIs.")
            
            # --- Integration Point ---
            # Here you will iterate through your rois, slice st.session_state.image0, 
            # run the bar segmentation, compute CTF, and pass the arrays to your fitting module:
            # f_array, ctf_array = execute_your_bar_segmentation_logic(sub_images)
            # fit_results, model = fit_two_gauss_unsharp_mask_mtf0(f_array, ctf_array)
            
            st.success("MTF Execution routed.")

    # --- Execution Logic: CNR ---
    if compute_cnr:
        append_log("CNR calculation started...")
        
        # --- Integration Point ---
        # Run the polar unwrap and segmentation
        # l_cart = wedge_segmentation(st.session_state.image0, st.session_state.info['px'])
        # Pass coordinates to calculate_cnr(...)
        
        append_log("CNR calculation completed.")
        st.success("CNR Execution routed.")

    # --- Results Export ---
    st.subheader("Data Export")
    if st.session_state.results:
        # Create an in-memory Excel file using Pandas
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            if 'MTFOutput' in st.session_state.results:
                df_mtf = pd.DataFrame(st.session_state.results['MTFOutput'], 
                                      columns=['Frequency (lp/mm)', 'CTF', 'MTF'])
                df_mtf.to_excel(writer, sheet_name='MTFOutput', index=False)
            
            if 'LargeObject' in st.session_state.results:
                df_large = pd.DataFrame(st.session_state.results['LargeObject'], 
                                        columns=['ROI mean GL', 'Background mean GL', 'Background Noise', 'CNR'])
                df_large.to_excel(writer, sheet_name='LargeObject', index=False)
                
            if 'SmallObject' in st.session_state.results:
                df_small = pd.DataFrame(st.session_state.results['SmallObject'], 
                                        columns=['ROI mean GL', 'Wedge mean GL', 'Background Noise', 'CNR'])
                df_small.to_excel(writer, sheet_name='SmallObject', index=False)
                
        # Format filename
        default_name = f"{st.session_state.info.get('PatientID', 'ID')}_{st.session_state.info.get('StudyDate', 'DATE')}_Results.xlsx"
        
        st.download_button(
            label="Download Results (XLSX)",
            data=buffer,
            file_name=default_name.upper(),
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary"
        )
