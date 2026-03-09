import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import torch
import SimpleITK as sitk
import numpy as np
from tqdm import tqdm
import logging
import sys

# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.unet_model import FusionUNet

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/inference.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class InferenceAgent:
    def __init__(self, config_path="configs/config.json", model_name="unet_best.pth"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        
        self.processed_dir = self.config["processed_dir"]
        self.models_dir = self.config["models_dir"]
        self.patch_size = np.array(self.config["patch_generation"]["patch_size"])
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model
        self.model = FusionUNet(in_channels=2, out_channels=1, init_features=16).to(self.device)
        model_path = os.path.join(self.models_dir, model_name)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            print("Model not found. Please train first.")
            logging.warning("Model not found.")

    def predict_sliding_window(self, ct_vol, pet_vol, patch_size, stride):
        """Performs sliding window inference."""
        z_dim, y_dim, x_dim = ct_vol.shape
        pz, py, px = patch_size
        sz, sy, sx = stride
        
        # Output volume
        prob_map = np.zeros(ct_vol.shape, dtype=np.float32)
        count_map = np.zeros(ct_vol.shape, dtype=np.float32)
        
        z_steps = range(0, z_dim - pz + sz, sz)
        y_steps = range(0, y_dim - py + sy, sy)
        x_steps = range(0, x_dim - px + sx, sx)
        
        # Fix steps to not overflow
        # If last step > dim, we take the last patch aligned to border
        
        for z in z_steps:
             if z + pz > z_dim: z = z_dim - pz
             for y in y_steps:
                 if y + py > y_dim: y = y_dim - py
                 for x in x_steps:
                     if x + px > x_dim: x = x_dim - px
                     
                     ct_patch = ct_vol[z:z+pz, y:y+py, x:x+px]
                     pet_patch = pet_vol[z:z+pz, y:y+py, x:x+px]
                     
                     # Check shape
                     if ct_patch.shape != tuple(patch_size): continue
                     
                     # Normalize (Same as PatchGenerator)
                     ct_norm = np.clip(ct_patch, -1000, 400)
                     ct_norm = (ct_norm + 1000) / 1400.0
                     
                     pet_norm = np.clip(pet_patch, 0, 20)
                     pet_norm = pet_norm / 20.0
                     
                     # Prepare Tensor
                     input_tensor = np.stack([ct_norm, pet_norm], axis=0)
                     input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(self.device)
                     
                     # Inference
                     with torch.no_grad():
                         with torch.cuda.amp.autocast():
                             output = self.model(input_tensor)
                         if torch.cuda.is_available():
                             torch.cuda.empty_cache() # Aggressive clearing
                         
                         prob = torch.sigmoid(output).cpu().numpy()[0, 0]
                         
                     # Accumulate
                     prob_map[z:z+pz, y:y+py, x:x+px] += prob
                     count_map[z:z+pz, y:y+py, x:x+px] += 1
                     
        # Average
        prob_map /= np.maximum(count_map, 1)
        return prob_map

    def run_inference(self, patient_id):
        patient_path = os.path.join(self.processed_dir, patient_id)
        ct_path = os.path.join(patient_path, "ct_cropped.nii.gz") # Use cropped
        pet_path = os.path.join(patient_path, "pet_cropped.nii.gz")
        
        if not os.path.exists(ct_path) or not os.path.exists(pet_path):
            return

        print(f"Running inference on {patient_id}...")
        
        ct_img = sitk.ReadImage(ct_path)
        pet_img = sitk.ReadImage(pet_path)
        
        ct_arr = sitk.GetArrayFromImage(ct_img)
        pet_arr = sitk.GetArrayFromImage(pet_img)
        
        # Padding if smaller than patch size
        # Handling small volumes is tricky, need padding.
        # Minimal implementation for now assumes > 64^3
        
        if np.any(np.array(ct_arr.shape) < self.patch_size):
            logging.warning(f"Volume too small for patch size in {patient_id}")
            return
            
        prob_map = self.predict_sliding_window(ct_arr, pet_arr, self.patch_size, stride=self.patch_size//2)
        
        # Threshold
        mask = (prob_map > 0.5).astype(np.uint8)
        
        # Post-processing: Remove small false positives using CCA
        if np.sum(mask) > 0:
            mask_sitk = sitk.GetImageFromArray(mask)
            cc_filter = sitk.ConnectedComponentImageFilter()
            labeled_mask = cc_filter.Execute(mask_sitk)
            
            relabel_filter = sitk.RelabelComponentImageFilter()
            relabel_filter.SortByObjectSizeOn()
            labeled_mask = relabel_filter.Execute(labeled_mask)
            
            # Keep only the largest component (label 1)
            mask = (sitk.GetArrayFromImage(labeled_mask) == 1).astype(np.uint8)
        
        # Save
        mask_out = sitk.GetImageFromArray(mask)
        mask_out.CopyInformation(ct_img)
        sitk.WriteImage(mask_out, os.path.join(patient_path, "prediction.nii.gz"))
        
        # Save Prob map too?
        # sitk.WriteImage(sitk.GetImageFromArray(prob_map), os.path.join(patient_path, "probability.nii.gz"))
        
        logging.info(f"Inference complete for {patient_id}")

    def run_batch(self):
        patients = [d for d in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, d))]
        for pid in tqdm(patients, desc="Inference"):
            self.run_inference(pid)

if __name__ == "__main__":
    agent = InferenceAgent()
    agent.run_batch()
