import os
import json
import SimpleITK as sitk
import numpy as np
import logging
from tqdm import tqdm

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/roi_extraction.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class LungROIExtractionAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.processed_dir = self.config["processed_dir"]

    def extract_lung_roi(self, ct_image, padding=10):
        """Extracts the bounding box of the lungs."""
        # Threshold to get body/lungs (HU < -300 approx for lung, but simple thresholding works for body crop)
        # Better: Threshold -1000 to -400 for lung parenchyma
        
        # 1. Binary Threshold
        binary = sitk.BinaryThreshold(ct_image, lowerThreshold=-1000, upperThreshold=-400, insideValue=1, outsideValue=0)
        
        # 2. Morphological Closing to fill holes
        binary = sitk.BinaryMorphologicalClosing(binary, [5, 5, 5])
        
        # 3. Get Largest Connected Component (Assumes lungs are connected or largest air pockets)
        # Note: Lungs might be separated. We usually want the body mask or just crop to non-background.
        # Simple approach: Crop to non-air body.
        # Let's try to crop to the body content > -1000
        
        body_mask = sitk.BinaryThreshold(ct_image, lowerThreshold=-990, upperThreshold=3000, insideValue=1, outsideValue=0)
        
        # Compute bounding box
        label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
        label_shape_filter.Execute(body_mask)
        
        if label_shape_filter.GetNumberOfLabels() > 0:
            bbox = label_shape_filter.GetBoundingBox(1) # (x, y, z, w, h, d)
            
            # Add padding
            x, y, z, w, h, d = bbox
            size = ct_image.GetSize()
            
            new_x = max(0, x - padding)
            new_y = max(0, y - padding)
            new_z = max(0, z - padding)
            
            new_w = min(size[0] - new_x, w + 2*padding)
            new_h = min(size[1] - new_y, h + 2*padding)
            new_d = min(size[2] - new_z, d + 2*padding)
            
            return [new_x, new_y, new_z], [new_w, new_h, new_d]
            
        return None, None

    def crop_volume(self, image, start_index, size):
        return sitk.RegionOfInterest(image, size, start_index)

    def resample_mask_to_reference(self, mask, reference):
        """Resamples mask to match reference image geometry."""
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor) # Important for labels
        resampler.SetOutputPixelType(sitk.sitkUInt8)
        return resampler.Execute(mask)

    def process_patient(self, patient_id):
        patient_path = os.path.join(self.processed_dir, patient_id)
        ct_path = os.path.join(patient_path, "ct_resampled.nii.gz")
        pet_path = os.path.join(patient_path, "pet_aligned.nii.gz") # Use aligned PET
        
        if not os.path.exists(ct_path):
            logging.warning(f"No CT found for {patient_id}")
            return
            
        try:
            ct_img = sitk.ReadImage(ct_path)
            
            start, size = self.extract_lung_roi(ct_img)
            
            if start and size:
                # Crop CT
                ct_cropped = self.crop_volume(ct_img, start, size)
                sitk.WriteImage(ct_cropped, os.path.join(patient_path, "ct_cropped.nii.gz"))
                
                # Crop PET (if exists and aligned)
                if os.path.exists(pet_path):
                    pet_img = sitk.ReadImage(pet_path)
                    pet_cropped = self.crop_volume(pet_img, start, size)
                    sitk.WriteImage(pet_cropped, os.path.join(patient_path, "pet_cropped.nii.gz"))

                # Crop Mask (if exists) -> Resample first!
                mask_path = os.path.join(patient_path, "mask_original.nii.gz")
                if os.path.exists(mask_path):
                    mask_img = sitk.ReadImage(mask_path)
                    # Resample mask to match CT geometry
                    mask_resampled = self.resample_mask_to_reference(mask_img, ct_img)
                    # Save resampled mask for debug?
                    # sitk.WriteImage(mask_resampled, os.path.join(patient_path, "mask_resampled.nii.gz"))
                    
                    mask_cropped = self.crop_volume(mask_resampled, start, size)
                    sitk.WriteImage(mask_cropped, os.path.join(patient_path, "mask_cropped.nii.gz"))
                
                # Save ROI info
                with open(os.path.join(patient_path, "roi_metadata.json"), "w") as f:
                    json.dump({"roi_start": start, "roi_size": size}, f)
                    
                logging.info(f"ROI Extraction successful for {patient_id}")
            else:
                logging.warning(f"Could not extract ROI for {patient_id}")

        except Exception as e:
            logging.error(f"ROI extraction failed for {patient_id}: {e}")

    def run_batch(self):
        if not os.path.exists(self.processed_dir):
            return

        patients = [d for d in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, d))]
        for pid in tqdm(patients, desc="Extracting ROIs"):
            self.process_patient(pid)

if __name__ == "__main__":
    agent = LungROIExtractionAgent()
    agent.run_batch()
