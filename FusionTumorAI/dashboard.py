import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import SimpleITK as sitk
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Configure Streamlit page
st.set_page_config(page_title="Lung PET-CT Clinical System", layout="wide", page_icon="🫁")

st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-right: 5px solid #ff4b2b;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #111; }
    .metric-label { font-size: 14px; color: #666; text-transform: uppercase; }
    </style>
""", unsafe_allow_html=True)

# Define paths
PROCESSED_DIR = "data/processed"

def get_patients():
    if not os.path.exists(PROCESSED_DIR):
        return []
    patients = [d for d in os.listdir(PROCESSED_DIR) if os.path.isdir(os.path.join(PROCESSED_DIR, d))]
    return sorted(patients)

def load_volume(patient_id, vol_type):
    path = os.path.join(PROCESSED_DIR, patient_id, f"{vol_type}_cropped.nii.gz")
    if not os.path.exists(path):
        # Fallback to resampled if cropped doesn't exist yet
        path = os.path.join(PROCESSED_DIR, patient_id, f"{vol_type}_resampled.nii.gz")
        if not os.path.exists(path):
            # Try specific naming for masks
            if vol_type == "mask":
                path = os.path.join(PROCESSED_DIR, patient_id, "prediction.nii.gz")
    
    if os.path.exists(path):
        img = sitk.ReadImage(path)
        return sitk.GetArrayFromImage(img), img.GetSpacing()
    return None, None

def load_metrics(patient_id):
    path = os.path.join(PROCESSED_DIR, patient_id, "radiomics.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}

# ----------------- Sidebar -----------------
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/NASA_logo.svg/200px-NASA_logo.svg.png", width=100) # Placeholder or remove
st.sidebar.title("Clinical Navigator")

patients = get_patients()
if not patients:
    st.sidebar.warning("No processed data found. Please run the preprocessing pipeline.")
    st.stop()

selected_patient = st.sidebar.selectbox("Select Patient", patients)
st.sidebar.markdown("---")
st.sidebar.subheader("Visualization Settings")
suv_threshold = st.sidebar.slider("SUV Threshold (PET Overlay)", min_value=0.0, max_value=15.0, value=2.5, step=0.1)
show_tumor = st.sidebar.checkbox("Show Tumor Overlay (Red)", value=True)
show_lymph = st.sidebar.checkbox("Show Lymph Nodes (Orange)", value=True)

# Load data for selected patient
st.title(f"Diagnostic View: {selected_patient}")

metrics = load_metrics(selected_patient)

# ----------------- Clinical Metrics -----------------
if metrics:
    st.markdown("### Oncological Biomarkers")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("SUV_Max", 0):.2f}</div><div class="metric-label">SUV Max</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("SUV_Mean", 0):.2f}</div><div class="metric-label">SUV Mean</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("Volume_cm3", 0):.2f} cc</div><div class="metric-label">Tumor Vol</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("MTV_cm3", 0):.2f}</div><div class="metric-label">MTV (cc)</div></div>', unsafe_allow_html=True)
    with col5:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{metrics.get("TLG", 0):.2f}</div><div class="metric-label">TLG</div></div>', unsafe_allow_html=True)

# ----------------- Main View -----------------
tab1, tab2 = st.tabs(["2D Multi-Planar Reconstruction (MPR)", "3D Interactive Surgical Viewer"])

with tab1:
    st.markdown("### Synchronized Tri-Planar View")
    ct_arr, spacing = load_volume(selected_patient, "ct")
    pet_arr, _ = load_volume(selected_patient, "pet")
    mask_arr, _ = load_volume(selected_patient, "mask")
    
    if ct_arr is not None:
        z_dim, y_dim, x_dim = ct_arr.shape
        
        # Sliders for synchronized crosshair
        colA, colB, colC = st.columns(3)
        with colA: axial_idx = st.slider("Axial Slice (Z)", 0, z_dim-1, z_dim//2)
        with colB: coronal_idx = st.slider("Coronal Slice (Y)", 0, y_dim-1, y_dim//2)
        with colC: sagittal_idx = st.slider("Sagittal Slice (X)", 0, x_dim-1, x_dim//2)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
        
        # Helper to plot slices
        def plot_slice(ax, ct_slice, pet_slice, mask_slice, title):
            # CT
            ax.imshow(ct_slice, cmap='gray', vmin=-1000, vmax=400)
            
            # PET Overlay
            if pet_slice is not None and np.max(pet_slice) > 0:
                pet_masked = np.ma.masked_where(pet_slice < suv_threshold, pet_slice)
                ax.imshow(pet_masked, cmap='hot', alpha=0.4, vmin=0, vmax=15)
                
            # Segmentation Mask Overlay
            if mask_slice is not None and show_tumor:
                tumor_masked = np.ma.masked_where(mask_slice == 0, mask_slice)
                ax.imshow(tumor_masked, cmap='autumn', alpha=0.5, vmin=0, vmax=1) # Red-ish
                
            # Layout
            ax.set_title(title, color='white')
            ax.axis('off')
            
        # Extract slices
        axial_ct = ct_arr[axial_idx, :, :]
        coronal_ct = np.flipud(ct_arr[:, coronal_idx, :])
        sagittal_ct = np.flipud(ct_arr[:, :, sagittal_idx])
        
        axial_pet = pet_arr[axial_idx, :, :] if pet_arr is not None else None
        coronal_pet = np.flipud(pet_arr[:, coronal_idx, :]) if pet_arr is not None else None
        sagittal_pet = np.flipud(pet_arr[:, :, sagittal_idx]) if pet_arr is not None else None
        
        axial_mask = mask_arr[axial_idx, :, :] if mask_arr is not None else None
        coronal_mask = np.flipud(mask_arr[:, coronal_idx, :]) if mask_arr is not None else None
        sagittal_mask = np.flipud(mask_arr[:, :, sagittal_idx]) if mask_arr is not None else None
        
        plot_slice(axes[0], axial_ct, axial_pet, axial_mask, "Axial")
        axes[0].axhline(y=coronal_idx, color='g', linestyle='--', alpha=0.5)
        axes[0].axvline(x=sagittal_idx, color='b', linestyle='--', alpha=0.5)
        
        plot_slice(axes[1], coronal_ct, coronal_pet, coronal_mask, "Coronal")
        axes[1].axhline(y=z_dim - axial_idx - 1, color='r', linestyle='--', alpha=0.5)
        axes[1].axvline(x=sagittal_idx, color='b', linestyle='--', alpha=0.5)
        
        plot_slice(axes[2], sagittal_ct, sagittal_pet, sagittal_mask, "Sagittal")
        axes[2].axhline(y=z_dim - axial_idx - 1, color='r', linestyle='--', alpha=0.5)
        axes[2].axvline(x=coronal_idx, color='g', linestyle='--', alpha=0.5)
        
        st.pyplot(fig)
    else:
        st.warning("Volume data not fully processed or missing for this patient.")

with tab2:
    st.markdown("### 3D Tumor Reconstruction")
    html_path = os.path.join(PROCESSED_DIR, selected_patient, "3d_viewer.html")
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            html_data = f.read()
        
        # Display the HTML in an iframe
        components.html(html_data, height=800, scrolling=False)
    else:
        st.info("Interactive 3D viewer not available. Run the visualization agent to generate it.")
