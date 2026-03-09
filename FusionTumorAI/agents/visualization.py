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
        
    def create_3d_snapshot(self, patient_id):
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
            
            # Create Plotter
            plotter = pv.Plotter(off_screen=True)
            
            # Lung/Body Volume (Ghost Lung) - using polygon mesh for stable headless rendering
            grid = pv.wrap(ct_arr)
            lung_hu_mask = np.logical_and(ct_arr > -1000, ct_arr < -400).astype(np.uint8)
            if np.sum(lung_hu_mask) > 0:
                lung_grid = pv.wrap(lung_hu_mask)
                lung_mesh = lung_grid.contour([0.5])
                plotter.add_mesh(lung_mesh, color="#e0e0e0", opacity=0.10, label="Lungs")
            
            # Tumor Surface (Red)
            # Tumor Surface (Intelligent Shadow)
            if np.sum(mask_arr) > 0:
                mask_grid = pv.wrap(mask_arr)
                tumor_mesh = mask_grid.contour([0.5])
                
                # Probe PET values for Color Mapping
                pet_grid = pv.wrap(pet_arr)
                # Sample PET at tumor mesh vertices. It inherits the array name from pet_grid (default is often 'values' or similar)
                tumor_mesh = tumor_mesh.sample(pet_grid)
                
                # PyVista assigns the sampled array name usually based on the source array name. Let's explicitly rename or find the active scalar.
                # The active scalar is what gets plotted.
                scalar_name = tumor_mesh.active_scalars_name
                if not scalar_name:
                    # fallback if no active scalars
                    scalar_name = tumor_mesh.array_names[0] if tumor_mesh.array_names else None
                
                # Add mesh with scalar mapping (SUV heatmap overlay)
                if scalar_name:
                    plotter.add_mesh(tumor_mesh, scalars=scalar_name, cmap="hot", opacity=1.0, label="Tumor (SUV Map)")
                    plotter.add_scalar_bar(title="SUV Intensity", n_labels=5)
                else:
                    plotter.add_mesh(tumor_mesh, color="red", opacity=1.0, label="Tumor")
                
                # Add 3D bounding box showing RECIST diameters
                bounds = tumor_mesh.bounds
                box = pv.Box(bounds)
                plotter.add_mesh(box, color="yellow", style="wireframe", line_width=2, label="RECIST Bounding Box")
            else:
                 pass
            
            # Add measurements/scale (map-like dimensions)
            plotter.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)', color='black', font_size=12)
            
            plotter.view_isometric()
            
            # Save Static Snapshot
            out_png = os.path.join(patient_path, "3d_render.png")
            plotter.screenshot(out_png)
            
            # 1. Save Raw GLTF First with Dual-Mesh and Vertex Colors
            out_gltf = os.path.join(patient_path, "tumor_mesh.gltf")
            gltf_saved = False
            if np.sum(mask_arr) > 0:
                try:
                    mesh_plotter = pv.Plotter(off_screen=True)
                    
                    # Extract Lung Mesh (Ghost Shell)
                    lung_hu_mask = np.logical_and(ct_arr > -1000, ct_arr < -400).astype(np.uint8)
                    if np.sum(lung_hu_mask) > 0:
                        lung_grid = pv.wrap(lung_hu_mask)
                        lung_mesh = lung_grid.contour([0.5])
                        # Simplify mesh to keep web viewer lightweight
                        lung_mesh = lung_mesh.decimate(0.95)
                        mesh_plotter.add_mesh(lung_mesh, color="#e0e0e0", opacity=0.10)
                    
                    # Bake Metabolic Color into Tumor Mesh
                    if scalar_name:
                        import matplotlib.pyplot as plt
                        pet_vals = tumor_mesh[scalar_name]
                        pet_max, pet_min = np.max(pet_vals), np.min(pet_vals)
                        diagnosis = "Benign/Indeterminate"
                        diagnosis_class = "benign-indeterminate"
                        if suv_max > 2.5:
                            if mean_hu > -400:
                                diagnosis = "Malignant Suspicion (Adenocarcinoma)"
                                diagnosis_class = "malignant-suspicion"
                            elif -900 <= mean_hu <= -500:
                                diagnosis = "Potential Infection (TB/Pneumonia)"
                                diagnosis_class = "potential-infection"
                            else:
                                diagnosis = "Suspicious (Uncertain Etiology)"
                                diagnosis_class = "suspicious"
                        if pet_max > pet_min:
                            norm_pet = (pet_vals - pet_min) / (pet_max - pet_min)
                        else:
                            norm_pet = np.zeros_like(pet_vals)
                        cmap = plt.get_cmap('hot')
                        colors = cmap(norm_pet)
                        tumor_mesh['RGB_colors'] = (colors[:, :3] * 255).astype(np.uint8)
                        tumor_mesh.active_scalars_name = 'RGB_colors'
                        mesh_plotter.add_mesh(tumor_mesh, scalars='RGB_colors', rgb=True)
                    else:
                        mesh_plotter.add_mesh(tumor_mesh, color="red")
                        
                    mesh_plotter.export_gltf(out_gltf)
                    mesh_plotter.close()
                    gltf_saved = True
                    
                    # --- FIX ALPHA MODE FOR TRANSPARENCY IN LOCAL HTML VIEWER ---
                    # PyVista does not write "alphaMode": "BLEND" to materials out of the box
                    # We must inject it into the raw JSON so Model-Viewer respects the transparency
                    import json
                    with open(out_gltf, 'r') as f:
                        gltf_data = json.load(f)

                    # Ensure all materials have alpha blending for transparency
                    if "materials" in gltf_data:
                        for mat in gltf_data["materials"]:
                            mat["alphaMode"] = "BLEND"
                            mat["doubleSided"] = True
                    
                    # Write it back out before it gets base64 encoded
                    with open(out_gltf, 'w') as f:
                        json.dump(gltf_data, f)
                        
                except Exception as e:
                     logging.error(f"GLTF export failed for {patient_id}: {e}")

            # 2. Save Interactive Web Viewer (NASA style) using <model-viewer>
            # To avoid CORS issues when opening local HTML files, we encode the GLTF as base64
            # and embed it directly into the src attribute.
            out_html = os.path.join(patient_path, "3d_viewer.html")
            try:
                gltf_src = "tumor_mesh.gltf" # fallback
                if gltf_saved and os.path.exists(out_gltf):
                    import base64
                    with open(out_gltf, "rb") as gf:
                        b64_data = base64.b64encode(gf.read()).decode('utf-8')
                        gltf_src = f"data:model/gltf+json;base64,{b64_data}"

                # Load dynamic metrics
                metrics = {}
                json_path = os.path.join(patient_path, "radiomics.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        metrics = json.load(f)

                suv_max = metrics.get('PET_original_firstorder_Maximum', 'N/A')
                if isinstance(suv_max, (int, float)):
                    suv_max = f"{suv_max:.2f}"
                tumor_vol = metrics.get('Tumor_Volume_cm3', 'N/A')
                if isinstance(tumor_vol, (int, float)):
                    tumor_vol = f"{tumor_vol:.2f}"
                sphericity = metrics.get('CT_original_shape_Sphericity', 'N/A')
                if isinstance(sphericity, (int, float)):
                    sphericity = f"{sphericity:.2f}"

                # Diagnosis logic (Infection Guard)
                diagnosis = "Benign/Indeterminate"
                diagnosis_class = "benign-indeterminate"
                if isinstance(suv_max, str):
                    # convert if needed
                    try:
                        suv_val = float(suv_max)
                    except:
                        suv_val = 0.0
                else:
                    suv_val = float(suv_max)
                # mean_hu is needed – retrieve from metrics if present
                mean_hu = metrics.get('CT_original_firstorder_Mean', -1000.0)
                if isinstance(mean_hu, str):
                    try:
                        mean_hu = float(mean_hu)
                    except:
                        mean_hu = -1000.0
                if suv_val > 2.5:
                    if mean_hu > -400:
                        diagnosis = "Malignant Suspicion (Adenocarcinoma)"
                        diagnosis_class = "malignant-suspicion"
                    elif -900 <= mean_hu <= -500:
                        diagnosis = "Potential Infection (TB/Pneumonia)"
                        diagnosis_class = "potential-infection"
                    else:
                        diagnosis = "Suspicious (Uncertain Etiology)"
                        diagnosis_class = "suspicious"

                # Calculate centroid of the tumor for camera targeting
                tumor_centroid = tumor_mesh.center if np.sum(mask_arr) > 0 else (0,0,0)
                # the focal point for typical model-viewer is relative to the scene bounding box or absolute coordinates
                # Since GLTF preserves coordinate space, we can use the center directly.
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
        body {{
            margin: 0;
            padding: 0;
            background-color: #ffffff;
            color: #333333;
            font-family: 'Inter', sans-serif;
            height: 100vh;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
            animation: fadeIn 0.8s ease-out forwards;
            opacity: 0;
        }}
@keyframes fadeIn {{
    to {{ opacity: 1; }}
}}
.overlay {{
    box-shadow: 0 8px 24px rgba(0,0,0,0.15);
}}
        model-viewer {{
            width: 100vw;
            height: 100vh;
            outline: none;
            --poster-color: transparent;
        }}
        .overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 12px;
            padding: 20px 25px;
            max-width: 350px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
            pointer-events: none; /* Let clicks pass through to the model */
        }}
        .overlay h1 {{
            margin: 0 0 10px 0;
            font-size: 1.4rem;
            font-weight: 600;
            letter-spacing: 0.5px;
            color: #111111;
        }}
        .overlay p {{
            margin: 0;
            font-size: 0.9rem;
            font-weight: 400;
            line-height: 1.5;
            color: #555555;
        }}
        .badge {{
            display: inline-block;
            font-size: 0.75rem;
            font-weight: 600;
            padding: 4px 10px;
            border-radius: 20px;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: white;
        }}
.badge.malignant-suspicion {{ background: linear-gradient(135deg, #ff416c, #ff4b2b); }}
.badge.potential-infection {{ background: linear-gradient(135deg, #00c6ff, #0072ff); }}
.badge.benign-indeterminate {{ background: linear-gradient(135deg, #7f7fd5, #86a8e7, #91eae4); }}
        /* Hide the default AR button if not wanted or style it */
        .Hotspot {{
            background: #fff;
            border-radius: 32px;
            border: 0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.25);
            box-sizing: border-box;
            cursor: pointer;
            height: 24px;
            padding: 8px;
            position: relative;
            transition: opacity 0.3s;
            width: 24px;
        }}
    </style>
</head>
<body>
    <model-viewer 
        src="{gltf_src}" 
        camera-controls 
        auto-rotate 
        rotation-per-second="10deg"
        shadow-intensity="1.5" 
        shadow-softness="1"
        exposure="1.0" 
        environment-image="neutral"
        interaction-prompt="none"
        camera-target="{camera_tgt}"
        camera-orbit="0deg 75deg 105%"
        bounds="tight">
    </model-viewer>
    
    <div class="overlay">
        <div class="badge {diagnosis_class}">{diagnosis}</div>
        <h1>Patient ID: {patient_id}</h1>
        <p>Interactive 3D reconstruction from PET-CT fusion. Click and drag to examine morphological features. Scroll to zoom.</p>
        <div style="margin-top: 15px; background: rgba(0,0,0,0.05); padding: 10px; border-radius: 8px;">
            <strong>SUV Max:</strong> {suv_max} <br>
            <strong>Tumor Vol:</strong> {tumor_vol} cm³ <br>
            <strong>Sphericity:</strong> {sphericity}
        </div>
        <p style="margin-top: 10px; font-size: 0.8rem; color: #888;">Rendered by FusionTumorAI Engine</p>
    </div>
</body>
</html>
"""
                with open(out_html, "w") as f:
                    f.write(html_content.strip())
                logging.info(f"Interactive HTML (model-viewer) saved for {patient_id}")
            except Exception as e:
                logging.error(f"HTML export failed for {patient_id}: {e}")
            
            # Save GIF (Rotating)
            out_gif = os.path.join(patient_path, "3d_rotation.gif")
            
            # Create 360-degree rotating GIF locked onto tumor centroid
            if np.sum(mask_arr) > 0:
                plotter.camera.focal_point = tumor_mesh.center
            else:
                plotter.camera.focal_point = grid.center
            
            # Fixed radius (100 mm) around tumor centroid for a tumor‑centric orbit
            radius = 100.0
            center = tumor_mesh.center if np.sum(mask_arr) > 0 else grid.center
            path = plotter.generate_orbital_path(n_points=36, radius=radius, center=center, shift=0)
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
    agent.create_3d_snapshot("Lung_Dx-A0164")
