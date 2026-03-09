import os
import json
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import random
import logging

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/patch_generator.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class PatchGeneratorAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.processed_dir = self.config["processed_dir"]
        self.patches_dir = self.config["patches_dir"]
        self.patch_size = np.array(self.config["patch_generation"]["patch_size"])
        self.samples_per_patient = self.config["patch_generation"]["samples_per_patient"]
        
        for split in ['train', 'val']:
            os.makedirs(os.path.join(self.patches_dir, split), exist_ok=True)

    def generate_patches(self, patient_id, split='train'):
        patient_path = os.path.join(self.processed_dir, patient_id)
        ct_path = os.path.join(patient_path, "ct_resampled.nii.gz")
        pet_path = os.path.join(patient_path, "pet_aligned.nii.gz")
        mask_path = os.path.join(patient_path, "mask_original.nii.gz") # Assumes mask exists
        
        if not os.path.exists(ct_path) or not os.path.exists(pet_path):
            logging.warning(f"Missing modalities for {patient_id}")
            return

        ct_img = sitk.ReadImage(ct_path)
        pet_img = sitk.ReadImage(pet_path)
        
        ct_arr = sitk.GetArrayFromImage(ct_img)
        pet_arr = sitk.GetArrayFromImage(pet_img)
        
        has_mask = os.path.exists(mask_path)
        mask_arr = None
        if has_mask:
            mask_img = sitk.ReadImage(mask_path)
            mask_arr = sitk.GetArrayFromImage(mask_img)
        
        # Normalize (Simple Z-score or MinMax)
        # CT: Clip and scale
        ct_arr = np.clip(ct_arr, -1000, 400)
        ct_arr = (ct_arr - (-1000)) / (400 - (-1000))
        
        # PET: SU V min-max (approx)
        pet_arr = np.clip(pet_arr, 0, 20)
        pet_arr = pet_arr / 20.0
        
        img_shape = ct_arr.shape
        
        # Define ROI (processed lungs/body)
        # We can use the whole image but sample effectively
        
        patches_generated = 0
        centers = []
        
        if has_mask and np.sum(mask_arr) > 0:
            # Get tumor centers
            tumor_indices = np.argwhere(mask_arr > 0)
            
            # 50% Tumor Centered
            num_tumor = self.samples_per_patient // 2
            for _ in range(num_tumor):
                if len(tumor_indices) > 0:
                    center = tumor_indices[random.randint(0, len(tumor_indices)-1)]
                    centers.append(center)
                
            # 50% Random
            num_random = self.samples_per_patient - num_tumor
        else:
            num_random = self.samples_per_patient

        # Random sampling with safety check for small volumes
        for _ in range(num_random):
            # Ensure valid range
            z_min, z_max = self.patch_size[0]//2, img_shape[0]-self.patch_size[0]//2
            y_min, y_max = self.patch_size[1]//2, img_shape[1]-self.patch_size[1]//2
            x_min, x_max = self.patch_size[2]//2, img_shape[2]-self.patch_size[2]//2
            
            # If volume < patch size, pick center (padding handles it later)
            z_c = random.randint(z_min, max(z_min, z_max))
            y_c = random.randint(y_min, max(y_min, y_max))
            x_c = random.randint(x_min, max(x_min, x_max))
            
            centers.append([z_c, y_c, x_c])
                
        # Extract
        for i, center in enumerate(centers):
            z, y, x = center
            dz, dy, dx = self.patch_size // 2
            
            z_start = max(0, z - dz)
            z_end = min(img_shape[0], z + dz)
            y_start = max(0, y - dy)
            y_end = min(img_shape[1], y + dy)
            x_start = max(0, x - dx)
            x_end = min(img_shape[2], x + dx)
            
            # Extract with padding if out of bounds
            ct_patch = np.full(self.patch_size, -1000.0) # Background value
            pet_patch = np.zeros(self.patch_size)
            
            # Calculate overlap range
            # Image coords
            z_start_img = max(0, z - dz)
            z_end_img = min(img_shape[0], z + dz)
            y_start_img = max(0, y - dy)
            y_end_img = min(img_shape[1], y + dy)
            x_start_img = max(0, x - dx)
            x_end_img = min(img_shape[2], x + dx)
            
            # Patch coords
            z_start_patch = z_start_img - (z - dz)
            if z_start_patch < 0: z_start_patch = 0 # Should use max(0, ...) relative logic
            # Simplified: calculate valid slice in patch
            p_z_s = max(0, (self.patch_size[0]//2) - (z - z_start_img))
            p_y_s = max(0, (self.patch_size[1]//2) - (y - y_start_img))
            p_x_s = max(0, (self.patch_size[2]//2) - (x - x_start_img))
            
            # Dimensions of valid data
            d_d = z_end_img - z_start_img
            d_h = y_end_img - y_start_img
            d_w = x_end_img - x_start_img
            
            # Verify we don't exceed patch bounds
            if p_z_s + d_d > self.patch_size[0] or p_y_s + d_h > self.patch_size[1] or p_x_s + d_w > self.patch_size[2]:
                 # Fallback to simple padding (robust way)
                 temp_ct = ct_arr[z_start_img:z_end_img, y_start_img:y_end_img, x_start_img:x_end_img]
                 temp_pet = pet_arr[z_start_img:z_end_img, y_start_img:y_end_img, x_start_img:x_end_img]
                 
                 # Calc padding
                 pad_z = (0, self.patch_size[0] - temp_ct.shape[0])
                 pad_y = (0, self.patch_size[1] - temp_ct.shape[1])
                 pad_x = (0, self.patch_size[2] - temp_ct.shape[2])
                 
                 # This aligns to top-left, but ensures size matches. Center alignment is better but complex.
                 ct_patch = np.pad(temp_ct, (pad_z, pad_y, pad_x), constant_values=-1000.0)
                 pet_patch = np.pad(temp_pet, (pad_z, pad_y, pad_x), constant_values=0)
            else:
                 # Check shapes match
                 ct_patch[p_z_s:p_z_s+d_d, p_y_s:p_y_s+d_h, p_x_s:p_x_s+d_w] = ct_arr[z_start_img:z_end_img, y_start_img:y_end_img, x_start_img:x_end_img]
                 pet_patch[p_z_s:p_z_s+d_d, p_y_s:p_y_s+d_h, p_x_s:p_x_s+d_w] = pet_arr[z_start_img:z_end_img, y_start_img:y_end_img, x_start_img:x_end_img]
            
            # Stack: (Channels, D, H, W)
            # PyTorch expects (C, D, H, W)
            patch = np.stack([ct_patch, pet_patch], axis=0)
            
            # Label
            label_patch = np.zeros(self.patch_size)
            if has_mask:
                # Use same logic as image extraction to ensure alignment and size
                if p_z_s + d_d <= self.patch_size[0] and p_y_s + d_h <= self.patch_size[1] and p_x_s + d_w <= self.patch_size[2]:
                    label_patch[p_z_s:p_z_s+d_d, p_y_s:p_y_s+d_h, p_x_s:p_x_s+d_w] = mask_arr[z_start_img:z_end_img, y_start_img:y_end_img, x_start_img:x_end_img]
                else:
                    # Fallback padding if needed (though the if check above covers the standard case)
                    temp_mask = mask_arr[z_start_img:z_end_img, y_start_img:y_end_img, x_start_img:x_end_img]
                    pad_z = (0, self.patch_size[0] - temp_mask.shape[0])
                    pad_y = (0, self.patch_size[1] - temp_mask.shape[1])
                    pad_x = (0, self.patch_size[2] - temp_mask.shape[2])
                    label_patch = np.pad(temp_mask, (pad_z, pad_y, pad_x), constant_values=0)
            
            # Save
            np.save(os.path.join(self.patches_dir, split, f"{patient_id}_p{i}_prod.npy"), patch.astype(np.float32))
            np.save(os.path.join(self.patches_dir, split, f"{patient_id}_p{i}_label.npy"), label_patch.astype(np.float32))
            
            patches_generated += 1
            
        logging.info(f"Generated {patches_generated} patches for {patient_id}")

    def run_batch(self):
        if not os.path.exists(self.processed_dir):
            return

        patients = [d for d in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, d))]
        
        # Simple split 80/20
        random.shuffle(patients)
        split_idx = int(len(patients) * 0.8)
        train_patients = patients[:split_idx]
        val_patients = patients[split_idx:]
        
        for pid in tqdm(train_patients, desc="Train Patches"):
            self.generate_patches(pid, 'train')
            
        for pid in tqdm(val_patients, desc="Val Patches"):
            self.generate_patches(pid, 'val')

if __name__ == "__main__":
    agent = PatchGeneratorAgent()
    agent.run_batch()
