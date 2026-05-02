import os
import json
import base64
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import io
from tqdm import tqdm

config_path = "configs/config.json"
with open(config_path, "r") as f:
    config = json.load(f)
processed_dir = config["processed_dir"]

patients = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]

for pid in tqdm(patients, desc="Fast-Patching HTMLs"):
    patient_path = os.path.join(processed_dir, pid)
    
    ct_path = os.path.join(patient_path, "ct_cropped.nii.gz")
    mask_path = os.path.join(patient_path, "prediction.nii.gz")
    if not os.path.exists(mask_path):
        mask_path = os.path.join(patient_path, "mask_cropped.nii.gz")
        
    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        continue

    # Load NIfTI for slices
    ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
    mask_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    
    pet_path = os.path.join(patient_path, "pet_cropped.nii.gz")
    if os.path.exists(pet_path):
        pet_arr = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))
    else:
        pet_arr = np.zeros_like(ct_arr)
        
    # Load radiomics metrics
    metrics = {}
    json_path = os.path.join(patient_path, "radiomics.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            metrics = json.load(f)

    suv_max_raw = metrics.get('PET_original_firstorder_Maximum', 'N/A')
    try: suv_val = float(suv_max_raw)
    except: suv_val = 0.0
    suv_max_str = f"{suv_val:.2f}" if suv_val > 0 else "N/A"
    suv_flag = " ⚠ Hypermetabolic" if suv_val > 2.5 else ""
    suv_color = "#ff4d4d" if suv_val > 2.5 else "#ffcc00"
    
    tumor_vol_raw = metrics.get('Tumor_Volume_cm3', 'N/A')
    try: tumor_vol_str = f"{float(tumor_vol_raw):.2f}"
    except: tumor_vol_str = "N/A"
    
    sph_raw = metrics.get('CT_original_shape_Sphericity', 'N/A')
    try:
        sph_val = float(sph_raw)
        sphericity_str = f"{sph_val:.2f}"
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
        ent_flag = " ⚠ Heterogeneous" if ent_val > 4.5 else " ✓ Homogeneous"
        ent_color = "#ff9900" if ent_val > 4.5 else "#66ff99"
    except:
        entropy_str = "N/A"
        ent_flag = ""
        ent_color = "#8b949e"

    diagnosis = "Benign/Indeterminate"
    diagnosis_class = "benign-indeterminate"
    if suv_val > 2.5:
        if -500 <= mean_hu <= -200:
            diagnosis = "Potential Infection (TB/Pneumonia)"
            diagnosis_class = "potential-infection"
        elif mean_hu > -400:
            diagnosis = "Malignant Suspicion (Adenocarcinoma)"
            diagnosis_class = "malignant-suspicion"
        else:
            diagnosis = "Suspicious (Uncertain Etiology)"
            diagnosis_class = "suspicious"

    # Find Centroid and Slices
    if np.sum(mask_arr) > 0:
        z_indices, _, _ = np.where(mask_arr > 0)
        cz = int(np.mean(z_indices))
    else:
        cz = ct_arr.shape[0] // 2
        
    fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=100)
    axes[0].imshow(ct_arr[cz, :, :], cmap='bone')
    axes[0].set_title('CT Axial', color='white')
    axes[0].axis('off')
    axes[1].imshow(pet_arr[cz, :, :], cmap='inferno')
    axes[1].set_title('PET Axial', color='white')
    axes[1].axis('off')
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

    # Fix GLTF AlphaMode
    out_gltf = os.path.join(patient_path, "tumor_mesh.gltf")
    if os.path.exists(out_gltf):
        with open(out_gltf, 'r') as f:
            gltf_data = json.load(f)
        if "materials" in gltf_data:
            for mat in gltf_data["materials"]:
                pbr = mat.get("pbrMetallicRoughness", {})
                color_factor = pbr.get("baseColorFactor", [1,1,1,1])
                if len(color_factor) > 3 and color_factor[3] < 1.0:
                    mat["alphaMode"] = "BLEND"
                elif "alphaMode" in mat:
                    del mat["alphaMode"] # Remove explicit BLEND for opaque objects
                mat["doubleSided"] = True
        with open(out_gltf, 'w') as f:
            json.dump(gltf_data, f)
            
        with open(out_gltf, "rb") as gf:
            b64_data = base64.b64encode(gf.read()).decode('utf-8')
            gltf_src = f"data:model/gltf+json;base64,{b64_data}"
    else:
        gltf_src = "tumor_mesh.gltf"
        
    # Get RECIST dummy values (approximate or just read from bounds if needed)
    # Since we aren't loading PyVista bounds, we can estimate from mask_arr
    if np.sum(mask_arr) > 0:
        z, y, x = np.where(mask_arr > 0)
        # Using 1mm spacing assumption for quick patch
        long_axis = float(max(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()))
        short_axis = float(min(x.max() - x.min(), y.max() - y.min(), z.max() - z.min()))
        camera_tgt = f"{(x.max()+x.min())/2}m {(y.max()+y.min())/2}m {(z.max()+z.min())/2}m"
    else:
        long_axis = 0.0
        short_axis = 0.0
        camera_tgt = "0m 0m 0m"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FusionTumorAI Interactive Viewer - {pid}</title>
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
        <h1>Patient ID: {pid}</h1>
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
                        if (color[3] < 1.0 || i === 0) {{
                            mat.pbrMetallicRoughness.setBaseColorFactor([color[0], color[1], color[2], opacity]);
                        }}
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>"""

    out_html = os.path.join(patient_path, "3d_viewer.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

print("Patching complete!")
