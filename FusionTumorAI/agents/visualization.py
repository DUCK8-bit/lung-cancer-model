import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import logging
import SimpleITK as sitk
import numpy as np
import pyvista as pv
from tqdm import tqdm

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/visualization.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class VisualizationAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.processed_dir = self.config["processed_dir"]
        self.reports_dir = self.config["reports_dir"]
        
        # PyVista settings for offscreen/headless if needed
        # pv.set_plot_theme("document")
        
    def create_3d_snapshot(self, patient_id, skip_gif=False):
        patient_path = os.path.join(self.processed_dir, patient_id)
        ct_path = os.path.join(patient_path, "ct_cropped.nii.gz")
        if os.path.exists(os.path.join(patient_path, "prediction.nii.gz")):
            mask_path = os.path.join(patient_path, "prediction.nii.gz")
        else:
            mask_path = os.path.join(patient_path, "mask_cropped.nii.gz")
        
        if not os.path.exists(ct_path) or not os.path.exists(mask_path): return

        try:
            # Load Data
            ct_img = sitk.ReadImage(ct_path)
            mask_img = sitk.ReadImage(mask_path)
            
            ct_arr = sitk.GetArrayFromImage(ct_img)
            mask_arr = sitk.GetArrayFromImage(mask_img)
            
            # Load PET for SUV mapping overlay
            pet_path = os.path.join(patient_path, "pet_cropped.nii.gz")
            if os.path.exists(pet_path):
                pet_img = sitk.ReadImage(pet_path)
                pet_arr = sitk.GetArrayFromImage(pet_img)
            else:
                pet_arr = np.zeros_like(ct_arr)
            
            # Load dynamic metrics early for Data Population & Infection Guard
            metrics = {}
            json_path = os.path.join(patient_path, "radiomics.json")
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    metrics = json.load(f)

            suv_max_raw = metrics.get('PET_original_firstorder_Maximum', 'N/A')
            try: suv_val = float(suv_max_raw)
            except: suv_val = 0.0
            suv_max_str = f"{suv_val:.2f}" if suv_val > 0 else "N/A"
            # Clinical threshold: SUV Max > 2.5 → hypermetabolic (malignant indicator)
            suv_flag = " ⚠ Hypermetabolic" if suv_val > 2.5 else ""
            suv_color = "#ff4d4d" if suv_val > 2.5 else "#ffcc00"
            
            tumor_vol_raw = metrics.get('Tumor_Volume_cm3', 'N/A')
            try: tumor_vol_str = f"{float(tumor_vol_raw):.2f}"
            except: tumor_vol_str = "N/A"
            
            sph_raw = metrics.get('CT_original_shape_Sphericity', 'N/A')
            try:
                sph_val = float(sph_raw)
                sphericity_str = f"{sph_val:.2f}"
                # Clinical threshold: Sphericity < 0.75 → irregular border (invasive)
                sph_flag = " ⚠ Irregular" if sph_val < 0.75 else " ✓ Regular"
                sph_color = "#ff9900" if sph_val < 0.75 else "#66ff99"
            except:
                sphericity_str = "N/A"
                sph_flag = ""
                sph_color = "#8b949e"

            mean_hu_raw = metrics.get('CT_original_firstorder_Mean', -1000.0)
            try: mean_hu = float(mean_hu_raw)
            except: mean_hu = -1000.0
            
            ent_raw = metrics.get('CT_original_firstorder_Entropy', 'N/A')
            try:
                ent_val = float(ent_raw)
                entropy_str = f"{ent_val:.2f}"
                # Clinical threshold: Entropy > 4.5 → heterogeneous texture (poor prognosis)
                ent_flag = " ⚠ Heterogeneous" if ent_val > 4.5 else " ✓ Homogeneous"
                ent_color = "#ff9900" if ent_val > 4.5 else "#66ff99"
            except:
                entropy_str = "N/A"
                ent_flag = ""
                ent_color = "#8b949e"

            # Infection Guard Layer implementation
            diagnosis = "Benign/Indeterminate"
            diagnosis_class = "benign-indeterminate"
            mesh_base_color_cmap = "inferno" # Fire colormap
            
            if suv_val > 2.5:
                if -500 <= mean_hu <= -200:
                    diagnosis = "Potential Infection (TB/Pneumonia)"
                    diagnosis_class = "potential-infection"
                    mesh_base_color_cmap = "Blues"  # Color mask Blue
                elif mean_hu > -400:
                    diagnosis = "Malignant Suspicion (Adenocarcinoma)"
                    diagnosis_class = "malignant-suspicion"
                    mesh_base_color_cmap = "inferno"  # Color mask Red / Fire
                else:
                    diagnosis = "Suspicious (Uncertain Etiology)"
                    diagnosis_class = "suspicious"
            else:
                # If SUV <= 2.5 but high HU, still might be benign or unsure, keep default
                pass

            import scipy.ndimage as ndimage
            import torch
            from monai.transforms import KeepLargestConnectedComponent
            import matplotlib.pyplot as plt

            # Create Plotter
            plotter = pv.Plotter(off_screen=True)
            
            # Apply Threshold Masking: Dual-Mesh System
            grid = pv.wrap(ct_arr)
            
            # Mesh A (Lungs): opacity of 0.1 for HU values -1000 to -400
            # If CT array is normalized to 0-255 (min >= 0), scale threshold accordingly
            if ct_arr.min() >= 0 and ct_arr.max() <= 255:
                lung_vol = grid.threshold([5, 75])
            else:
                lung_vol = grid.threshold([-1000, -400])
                
            if lung_vol.n_points > 0:
                lung_mesh = lung_vol.extract_surface()
                plotter.add_mesh(lung_mesh, color="#e0e0e0", opacity=0.10, label="Lungs")
            else:
                lung_mesh = None
            
            tumor_centroid = (0.0, 0.0, 0.0)
            tumor_mesh = None
            scalar_name = None

            if np.sum(mask_arr) > 0:
                mask_grid = pv.wrap(mask_arr)
                # Mesh B (Tumor): opacity 1.0
                tumor_mesh = mask_grid.contour([0.5])
                
                # Metabolic Color Baking
                pet_grid = pv.wrap(pet_arr)
                tumor_mesh = tumor_mesh.sample(pet_grid)
                
                scalar_name = tumor_mesh.active_scalars_name
                if not scalar_name and tumor_mesh.array_names:
                    scalar_name = tumor_mesh.array_names[0]
                
                if scalar_name:
                    pet_vals = tumor_mesh[scalar_name]
                    # Map SUV intensities
                    # Set tumor region to 1.0 opacity (solid mass)
                    plotter.add_mesh(tumor_mesh, scalars=scalar_name, cmap=mesh_base_color_cmap, opacity=1.0, label="Tumor (SUV Map)")
                    plotter.add_scalar_bar(title="SUV Intensity", n_labels=5)
                else:
                    plotter.add_mesh(tumor_mesh, color="red" if mesh_base_color_cmap == "inferno" else "blue", opacity=1.0, label="Tumor")
                
                # Bounding box showing RECIST diameters
                bounds = tumor_mesh.bounds
                box = pv.Box(bounds)
                plotter.add_mesh(box, color="yellow", style="wireframe", line_width=2, label="RECIST Bounding Box")
                
                tumor_centroid = tumor_mesh.center
                
                # Automated RECIST Measurements
                dx = bounds[1] - bounds[0]
                dy = bounds[3] - bounds[2]
                dz = bounds[5] - bounds[4]
                dims = sorted([dx, dy, dz], reverse=True)
                long_axis = dims[0]
                short_axis = dims[1]
                
                plotter.add_point_labels(
                    [tumor_centroid],
                    [f"RECIST:\nL: {long_axis:.1f}mm\nS: {short_axis:.1f}mm"],
                    text_color="white",
                    font_size=18,
                    shape="rounded_rect",
                    shape_opacity=0.6,
                    always_visible=True
                )
            else:
                tumor_centroid = grid.center
                long_axis = 0.0
                short_axis = 0.0
            
            # Map-like coordinates for reference
            plotter.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', color='black', font_size=12)
            plotter.view_isometric()
            
            # Static Snapshot
            out_png = os.path.join(patient_path, "3d_render.png")
            plotter.screenshot(out_png)
            
            # GLTF Export (dual mesh with baked vertex colors)
            out_gltf = os.path.join(patient_path, "tumor_mesh.gltf")
            gltf_saved = False
            if tumor_mesh is not None:
                try:
                    mesh_plotter = pv.Plotter(off_screen=True)
                    
                    if lung_mesh is not None:
                        lung_mesh_gltf = lung_mesh.triangulate().decimate(0.95)
                        mesh_plotter.add_mesh(lung_mesh_gltf, color="#e0e0e0", opacity=0.03) # Lower opacity for HTML viewer clarity
                    
                    # Ensure tumor is a distinct red mesh for the 3D model viewer
                    if mesh_base_color_cmap == "Blues":
                        tumor_color = "blue"
                    else:
                        tumor_color = "red"
                        
                    mesh_plotter.add_mesh(tumor_mesh, color=tumor_color, opacity=1.0)
                        
                    mesh_plotter.export_gltf(out_gltf)
                    mesh_plotter.close()
                    gltf_saved = True
                    
                    # Fix alpha mode for transparency in model-viewer
                    with open(out_gltf, 'r') as f:
                        gltf_data = json.load(f)
                    if "materials" in gltf_data:
                        for mat in gltf_data["materials"]:
                            # Only set BLEND for transparent materials to avoid depth-sorting issues
                            pbr = mat.get("pbrMetallicRoughness", {})
                            color_factor = pbr.get("baseColorFactor", [1,1,1,1])
                            if len(color_factor) > 3 and color_factor[3] < 1.0:
                                mat["alphaMode"] = "BLEND"
                            mat["doubleSided"] = True
                    with open(out_gltf, 'w') as f:
                        json.dump(gltf_data, f)
                except Exception as e:
                    logging.error(f"GLTF export failed for {patient_id}: {e}")

            import io
            import base64
            # Create slices for cross-reference
            if np.sum(mask_arr) > 0:
                z_indices, _, _ = np.where(mask_arr > 0)
                cz = int(np.mean(z_indices))
            else:
                cz = ct_arr.shape[0] // 2
                
            fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=100)
            
            # CT Slice
            axes[0].imshow(ct_arr[cz, :, :], cmap='bone')
            axes[0].set_title('CT Axial', color='white')
            axes[0].axis('off')
            
            # PET Slice
            axes[1].imshow(pet_arr[cz, :, :], cmap='inferno')
            axes[1].set_title('PET Axial', color='white')
            axes[1].axis('off')
            
            # Fusion/Mask
            axes[2].imshow(ct_arr[cz, :, :], cmap='bone')
            axes[2].imshow(mask_arr[cz, :, :], cmap='Reds', alpha=0.5)
            axes[2].set_title('Segmentation', color='white')
            axes[2].axis('off')
            
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight', facecolor='#161b22')
            plt.close(fig)
            slices_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            slices_html = f'<div class="slices-panel"><div style="font-size:0.8rem; font-weight:600; margin-bottom:8px; color:#e6edf3;">Cross-Reference Slices (Z={cz})</div><img src="data:image/png;base64,{slices_b64}" style="width: 100%; border-radius: 8px;"/></div>'

            # Save Interactive Web Viewer
            out_html = os.path.join(patient_path, "3d_viewer.html")
            try:
                gltf_src = "tumor_mesh.gltf"
                if gltf_saved and os.path.exists(out_gltf):
                    with open(out_gltf, "rb") as gf:
                        b64_data = base64.b64encode(gf.read()).decode('utf-8')
                        gltf_src = f"data:model/gltf+json;base64,{b64_data}"

                camera_tgt = f"{tumor_centroid[0]}m {tumor_centroid[1]}m {tumor_centroid[2]}m"

                html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FusionTumorAI Interactive Viewer - {patient_id}</title>
    <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.4.0/model-viewer.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        body {{ margin: 0; padding: 0; background-color: #0d1117; color: #e6edf3; font-family: 'Inter', sans-serif; height: 100vh; overflow: hidden; display: flex; justify-content: center; align-items: center; animation: fadeIn 0.8s ease-out forwards; opacity: 0; }}
        @keyframes fadeIn {{ to {{ opacity: 1; }} }}
        model-viewer {{ width: 100vw; height: 100vh; outline: none; --poster-color: transparent; }}
        .overlay {{ position: absolute; top: 20px; left: 20px; background: rgba(22, 27, 34, 0.85); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 20px 25px; max-width: 350px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5); pointer-events: none; }}
        .overlay h1 {{ margin: 0 0 10px 0; font-size: 1.4rem; font-weight: 600; color: #ffffff; }}
        .overlay p {{ margin: 0; font-size: 0.9rem; font-weight: 400; line-height: 1.5; color: #8b949e; }}
        .badge {{ display: inline-block; font-size: 0.75rem; font-weight: 600; padding: 4px 10px; border-radius: 20px; margin-bottom: 12px; text-transform: uppercase; color: white; }}
        .badge.malignant-suspicion {{ background: linear-gradient(135deg, #ff416c, #ff4b2b); }}
        .badge.potential-infection {{ background: linear-gradient(135deg, #00c6ff, #0072ff); }}
        .badge.benign-indeterminate {{ background: linear-gradient(135deg, #7f7fd5, #86a8e7, #91eae4); }}
        .badge.suspicious {{ background: linear-gradient(135deg, #f7971e, #ffd200); }}
        .control-panel {{ position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); background: rgba(22, 27, 34, 0.85); padding: 15px 25px; border-radius: 12px; display: flex; gap: 15px; align-items: center; color: white; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }}
        .control-panel input[type=range] {{ width: 150px; cursor: pointer; accent-color: #0072ff; }}
        .slices-panel {{ position: absolute; bottom: 20px; right: 20px; background: rgba(22, 27, 34, 0.85); padding: 15px; border-radius: 12px; width: 350px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }}
    </style>
</head>
<body>
    <model-viewer 
        src="{gltf_src}" 
        camera-controls 
        auto-rotate 
        rotation-per-second="5deg"
        shadow-intensity="2.0" 
        shadow-softness="1"
        exposure="1.2" 
        environment-image="neutral"
        interaction-prompt="none"
        camera-target="{camera_tgt}"
        camera-orbit="45deg 75deg 40%"
        bounds="tight">
    </model-viewer>
    
    <div class="overlay">
        <div class="badge {diagnosis_class}">{diagnosis}</div>
        <h1>Patient ID: {patient_id}</h1>
        <p>Interactive 3D reconstruction from PET-CT fusion. Click and drag to examine morphological features. Scroll to zoom.</p>
        <div style="margin-top: 15px; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; font-size: 0.85rem; line-height: 1.9;">
            <strong>SUV Max:</strong> <span style="color:{suv_color}">{suv_max_str}{suv_flag}</span> <br>
            <strong>Tumor Vol:</strong> {tumor_vol_str} cm³ <br>
            <strong>Sphericity:</strong> <span style="color:{sph_color}">{sphericity_str}{sph_flag}</span> <br>
            <strong>Entropy:</strong> <span style="color:{ent_color}">{entropy_str}{ent_flag}</span> <br>
            <strong>RECIST Ruler:</strong> {long_axis:.1f} x {short_axis:.1f} mm
        </div>
        <p style="margin-top: 10px; font-size: 0.8rem; color: #6e7681;">Rendered by FusionTumorAI Engine</p>
    </div>

    <div class="control-panel">
        <span style="font-size: 0.9rem; font-weight: 600;">Ghost Mesh (Lung)</span>
        <input type="range" id="opacity-slider" min="0" max="0.3" step="0.01" value="0.03">
    </div>

    {slices_html}

    <script>
        document.addEventListener("DOMContentLoaded", () => {{
            const viewer = document.querySelector('model-viewer');
            const slider = document.getElementById('opacity-slider');
            
            slider.addEventListener('input', (e) => {{
                const opacity = parseFloat(e.target.value);
                if (viewer.model && viewer.model.materials) {{
                    for (let i=0; i<viewer.model.materials.length; i++) {{
                        const mat = viewer.model.materials[i];
                        const color = mat.pbrMetallicRoughness.baseColorFactor;
                        // Target the lung mesh material (initial opacity < 1.0)
                        if (color[3] < 1.0 || i === 0) {{
                            mat.pbrMetallicRoughness.setBaseColorFactor([color[0], color[1], color[2], opacity]);
                        }}
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>
"""
                with open(out_html, "w", encoding="utf-8") as f:
                    f.write(html_content.strip())
                logging.info(f"Interactive HTML (model-viewer) saved for {patient_id}")
            except Exception as e:
                logging.error(f"HTML export failed for {patient_id}: {e}")
            
            # Save GIF (Rotating) - skip for very large volumes to prevent timeout
            voxel_count = ct_arr.size
            if skip_gif or voxel_count > 5_000_000:
                logging.info(f"Skipping GIF for {patient_id} (voxels={voxel_count}, skip_gif={skip_gif})")
                plotter.close()
            else:
                out_gif = os.path.join(patient_path, "3d_rotation.gif")
                
                # Centroid-Locked Rotation: Set camera focal point to tumor centroid exactly
                plotter.camera.focal_point = tumor_centroid
                
                # Reduce GIF radius to focus tightly on the pathology by adjusting factor
                factor = 1.2
                path = plotter.generate_orbital_path(n_points=18, factor=factor, shift=0)
                try:
                    plotter.open_gif(out_gif)
                    plotter.orbit_on_path(path, write_frames=True)
                except Exception as e:
                    logging.error(f"GIF export failed: {e}")
                finally:
                    plotter.close()
            
            logging.info(f"Visualizations created for {patient_id}")
            
        except Exception as e:
            logging.error(f"Visualization failed for {patient_id}: {e}")

    def run_batch(self):
        patients = [d for d in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, d))]
        for pid in tqdm(patients, desc="Visualizing"):
            self.create_3d_snapshot(pid)

if __name__ == "__main__":
    agent = VisualizationAgent()
    print("Starting batch visualization for all patients...")
    agent.run_batch()
    print("Batch visualization complete!")
