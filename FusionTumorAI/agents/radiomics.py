import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import logging
import pandas as pd
import SimpleITK as sitk
import numpy as np
from scipy.stats import entropy as scipy_entropy
from tqdm import tqdm
try:
    from radiomics import featureextractor
    HAS_PYRADIOMICS = True
except ImportError:
    HAS_PYRADIOMICS = False
    logging.warning("pyradiomics not installed. Radiomics features will be skipped.")

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/radiomics.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RadiomicsAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.processed_dir = self.config["processed_dir"]
        
        # Configure Extractor
        if HAS_PYRADIOMICS:
            self.extractor = featureextractor.RadiomicsFeatureExtractor()
            # Enable features
            self.extractor.enableFeatureClassByName('firstorder')
            self.extractor.enableFeatureClassByName('shape')
            self.extractor.enableFeatureClassByName('glcm')
            self.extractor.enableFeatureClassByName('glrlm')
            self.extractor.enableFeatureClassByName('glszm')
        else:
            self.extractor = None

    def extract_features(self, patient_id):
        patient_path = os.path.join(self.processed_dir, patient_id)
        ct_path = os.path.join(patient_path, "ct_cropped.nii.gz")
        pet_path = os.path.join(patient_path, "pet_cropped.nii.gz")
        mask_path = os.path.join(patient_path, "prediction.nii.gz") # Use AI prediction
        
        # If prediction validation, we might want to use "mask.nii.gz" (Ground Truth) for training the classifier
        # Strategy: Use Ground Truth if available for Training the Classifier, use Prediction for Inference.
        # But for 'RadiomicsAgent' in the pipeline, it usually extracts from the available mask.
        # Let's prefer Ground Truth if available for creating the 'features.csv' dataset.
        
        # Prioritize prediction if it has data, otherwise ground truth
        gt_path = os.path.join(patient_path, "mask_cropped.nii.gz")
        
        target_mask = None
        if os.path.exists(mask_path):
            m_arr = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
            if np.sum(m_arr) > 0:
                target_mask = mask_path
                
        if not target_mask and os.path.exists(gt_path):
            target_mask = gt_path

        if not os.path.exists(ct_path) or not target_mask:
            return None

        features = {"PatientID": patient_id, "Source": "GT" if target_mask == gt_path else "Pred"}
        
        try:
            # Check if mask is empty
            mask = sitk.ReadImage(target_mask)
            mask_arr = sitk.GetArrayFromImage(mask)
            ct_arr = sitk.GetArrayFromImage(sitk.ReadImage(ct_path))
            
            pet_arr = None
            if os.path.exists(pet_path):
                pet_arr = sitk.GetArrayFromImage(sitk.ReadImage(pet_path))

            if np.sum(mask_arr) == 0:
                logging.warning(f"Empty mask for {patient_id}")
                return None
            
            # --- Native Metrics Calculation (No PyRadiomics Needed) ---
            import pyvista as pv
            # Volume & Sphericity using PyVista
            mask_grid = pv.wrap(mask_arr)
            tumor_mesh = mask_grid.contour([0.5])
            
            spacing = mask.GetSpacing()
            # Approximation of volume using voxel counting (Robust)
            voxel_count = np.sum(mask_arr > 0)
            volume_cm3 = (voxel_count * spacing[0] * spacing[1] * spacing[2]) / 1000.0
            
            # Surface area and exact mesh volume
            if tumor_mesh.n_points > 0:
                mesh_vol = tumor_mesh.volume * (spacing[0] * spacing[1] * spacing[2]) / 1000.0
                mesh_area = tumor_mesh.area * (spacing[0] * spacing[1]) # Approx area scaling
                
                # Sphericity = (pi^(1/3) * (6*V)^(2/3)) / A
                # Use voxel count volume for stability if mesh volume is tiny
                # Fix: use spacing[0]*spacing[1] for correct pixel area (not mean**2)
                if mesh_area > 0 and volume_cm3 > 0:
                    v_mm3 = volume_cm3 * 1000.0
                    a_mm2 = tumor_mesh.area * (spacing[0] * spacing[1])
                    sphericity = (np.pi**(1.0/3.0) * (6.0 * v_mm3)**(2.0/3.0)) / a_mm2
                    sphericity = float(np.clip(sphericity, 0.0, 1.0))  # bounded [0,1]
                else:
                    sphericity = 0.0
                
                # RECIST roughly
                bounds = tumor_mesh.bounds
                recist = np.max([bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4]]) * np.max(spacing)
            else:
                sphericity = 0.0
                recist = 0.0

            # HU Stats
            tumor_hu = ct_arr[mask_arr > 0]
            mean_hu = np.mean(tumor_hu) if len(tumor_hu) > 0 else -1000.0
            
            # SUV Max (non-negative; PET voxel values represent radiotracer uptake)
            suv_max = 0.0
            if pet_arr is not None:
                tumor_suv = pet_arr[mask_arr > 0]
                suv_max = float(np.max(tumor_suv)) if len(tumor_suv) > 0 else 0.0
                suv_max = max(0.0, suv_max)  # SUV is always non-negative
            else:
                logging.warning(f"No PET file for {patient_id} — SUV Max set to 0.0")

            # Entropy: Shannon entropy from CT intensity histogram within tumor mask
            # Higher entropy = more texture heterogeneity = poorer prognosis indicator
            entropy_val = 0.0
            if len(tumor_hu) > 0:
                hist, _ = np.histogram(tumor_hu, bins=64, range=(-1000, 1000), density=True)
                hist_pos = hist[hist > 0]  # avoid log(0)
                entropy_val = float(scipy_entropy(hist_pos, base=2))

            # Inject calculated values
            features["Tumor_Volume_cm3"] = float(volume_cm3)
            features["CT_original_shape_Sphericity"] = float(sphericity)
            features["CT_original_shape_Maximum3DDiameter"] = float(recist)
            features["CT_original_firstorder_Mean"] = float(mean_hu)
            features["PET_original_firstorder_Maximum"] = float(suv_max)
            features["CT_original_firstorder_Entropy"] = entropy_val
            
            # Extract PyRadiomics if available just to supplement
            if self.extractor:
                try:
                    ct_feats = self.extractor.execute(ct_path, target_mask)
                    for k, v in ct_feats.items():
                        if not k.startswith("diagnostics") and k not in features:
                            features[f"CT_{k}"] = v
                    if os.path.exists(pet_path):
                        pet_feats = self.extractor.execute(pet_path, target_mask)
                        for k, v in pet_feats.items():
                            if not k.startswith("diagnostics") and k not in features:
                                features[f"PET_{k}"] = v
                except:
                    pass
                        
            # Save per-patient JSON
            json_path = os.path.join(patient_path, "radiomics.json")
            def convert(o):
                if isinstance(o, np.int64): return int(o)
                if isinstance(o, np.float64): return float(o)
                if isinstance(o, np.ndarray): return o.tolist()
                return o
            
            with open(json_path, "w") as f:
                json.dump(features, f, default=convert, indent=4)
            
            return features
            
        except Exception as e:
            logging.error(f"Radiomics extraction failed for {patient_id}: {e}")
            return None

    def run_batch(self):
        patients = [d for d in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, d))]
        
        all_features = []
        for pid in tqdm(patients, desc="Extracting Radiomics"):
            feat = self.extract_features(pid)
            if feat:
                all_features.append(feat)
                
        if all_features:
            df = pd.DataFrame(all_features)
            out_file = os.path.join(self.config["models_dir"], "radiomics_features.csv")
            df.to_csv(out_file, index=False)
            print(f"Saved features to {out_file}")

if __name__ == "__main__":
    agent = RadiomicsAgent()
    agent.run_batch()
