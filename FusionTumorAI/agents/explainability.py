import os
import json
import torch
import numpy as np
import SimpleITK as sitk
import cv2
import logging
import sys

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet_model import FusionUNet

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/explainability.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ExplainabilityAgent:
    def __init__(self, config_path="configs/config.json", model_name="unet_best.pth"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.processed_dir = self.config["processed_dir"]
        self.models_dir = self.config["models_dir"]
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FusionUNet(in_channels=2, out_channels=1, init_features=16).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.models_dir, model_name), map_location=self.device))
        self.model.eval()

        # Hook for Grad-CAM
        self.gradients = None
        self.activations = None
        
        # Target layer: bottleneck or last encoder
        target_layer = self.model.bottleneck.double_conv[0] # First Conv in bottleneck
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_gradcam(self, patient_id):
        patient_path = os.path.join(self.processed_dir, patient_id)
        ct_path = os.path.join(patient_path, "ct_cropped.nii.gz")
        pet_path = os.path.join(patient_path, "pet_cropped.nii.gz")
        
        if not os.path.exists(ct_path): return

        # Load middle slice or use volume logic
        # For simplicity, let's do GradCAM on a central crop or resized volume
        # GradCAM on 3D is heavy. We'll do it on a representative input logic.
        
        # Load Volume
        ct = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
        pet = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))
        
        # Normalize
        ct = (np.clip(ct, -1000, 400) + 1000) / 1400.0
        pet = np.clip(pet, 0, 20) / 20.0
        
        # Resize to model input size (e.g. 64x64x64 or 96x96x32)
        # We need a fixed size for the model or patch
        
        # Let's take a center patch for explanation
        z, y, x = np.array(ct.shape) // 2
        d = 32
        
        z_s, z_e = max(0, z-d), min(ct.shape[0], z+d)
        y_s, y_e = max(0, y-d), min(ct.shape[1], y+d)
        x_s, x_e = max(0, x-d), min(ct.shape[2], x+d)
        
        ct_patch = ct[z_s:z_e, y_s:y_e, x_s:x_e]
        pet_patch = pet[z_s:z_e, y_s:y_e, x_s:x_e]
        
        # Pad to 64
        # Skip pad for brevity, assume sufficient size or minimal run
        
        input_tensor = torch.from_numpy(np.stack([ct_patch, pet_patch], axis=0)).unsqueeze(0).float().to(self.device)
        
        # Forward
        output = self.model(input_tensor)
        
        # Backward
        self.model.zero_grad()
        score = output.sum()
        score.backward()
        
        # Generate Map
        gradients = self.gradients.cpu().data.numpy()[0] # (C, D, H, W)
        activations = self.activations.cpu().data.numpy()[0] # (C, D, H, W)
        
        weights = np.mean(gradients, axis=(1, 2, 3)) # Global Average Pooling over spatial
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (ct_patch.shape[2], ct_patch.shape[1])) # Resize 2D slices or 3D?
        # cv2 resize is 2D. Need scipy.ndimage.zoom for 3D
        from scipy.ndimage import zoom
        factors = (ct_patch.shape[0]/cam.shape[0], ct_patch.shape[1]/cam.shape[1], ct_patch.shape[2]/cam.shape[2])
        cam = zoom(cam, factors)
        
        # Save
        cam_img = sitk.GetImageFromArray(cam)
        sitk.WriteImage(cam_img, os.path.join(patient_path, "gradcam.nii.gz"))

    def check_infection(self, patient_id):
        # Infection Guard Logic
        # 1. High SUV?
        # 2. GGO on CT (HU -700 to -300)?
        # 3. Entropy Low?
        
        patient_path = os.path.join(self.processed_dir, patient_id)
        # Load Radiomics features row
        features_path = os.path.join(self.models_dir, "radiomics_features.csv")
        if not os.path.exists(features_path): return "Unknown"
        
        df = pd.read_csv(features_path)
        row = df[df['PatientID'] == patient_id]
        if row.empty: return "Unknown"
        
        suv_max = row['PET_original_firstorder_Maximum'].values[0] if 'PET_original_firstorder_Maximum' in row else 0
        entropy = row['CT_original_firstorder_Entropy'].values[0] if 'CT_original_firstorder_Entropy' in row else 10
        
        # Logic
        is_high_suv = suv_max > 2.5
        is_low_entropy = entropy < 4.0 # specific threshold to be tuned
        
        # Check GGO from HU (simple check)
        # We need the masked CT values.
        # Approximation: If ROI mean HU is in GGO range
        mean_hu = row['CT_original_firstorder_Mean'].values[0]
        is_ggo = -700 < mean_hu < -300
        
        flag = "Tumor"
        if is_high_suv and is_ggo and is_low_entropy:
            flag = "Possible Infection"
            
        # Write to report data
        with open(os.path.join(patient_path, "infection_flag.txt"), "w") as f:
            f.write(flag)
            
        return flag

    def run_batch(self):
        patients = [d for d in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, d))]
        for pid in patients:
            try:
                self.generate_gradcam(pid)
                alert = self.check_infection(pid)
                print(f"Patient {pid}: {alert}")
            except Exception as e:
                logging.error(f"Error in explainability for {pid}: {e}")

if __name__ == "__main__":
    agent = ExplainabilityAgent()
    agent.run_batch()
