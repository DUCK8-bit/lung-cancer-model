import os
import json
import SimpleITK as sitk
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/preprocessing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DICOMPreprocessingAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.output_dir = self.config["processed_dir"]
        self.target_spacing = [1.0, 1.0, 1.0] # Enforce 1mm3 isotropic
        os.makedirs(self.output_dir, exist_ok=True)

    def resample_image(self, image, target_spacing, is_label=False):
        """Resamples an image to a new spacing."""
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        
        new_size = [
            int(round(osz * osp / nsp))
            for osz, osp, nsp in zip(original_size, original_spacing, target_spacing)
        ]
        
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(image.GetPixelIDValue())
        
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkBSpline)
            
        return resampler.Execute(image)

    def process_patient(self, patient_id, ct_series_path, pet_series_path):
        """Converts DICOM to NIfTI and resamples."""
        patient_out_dir = os.path.join(self.output_dir, patient_id)
        os.makedirs(patient_out_dir, exist_ok=True)
        
        try:
            # reader = sitk.ImageSeriesReader()
            
            # Process CT
            if ct_series_path:
                print(f"Processing CT for {patient_id}...")
                ct_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(ct_series_path)
                ct_img = sitk.ReadImage(ct_names)
                
                # Check for RGB/Vector (Secondary Capture)
                if ct_img.GetNumberOfComponentsPerPixel() > 1:
                    logging.warning(f"CT {patient_id} is Vector/RGB. Converting to Grayscale.")
                    ct_img = sitk.VectorIndexSelectionCast(ct_img, 0)
                
                # Resample
                ct_resampled = self.resample_image(ct_img, self.target_spacing)
                
                # Apply Clinical HU Windowing (Lung/Mediastinal combined clinical bounds)
                # Lung: WL -600, WW 1500 -> [-1350, 150]
                # Mediastinal: WL 40, WW 400 -> [-160, 240]
                # Bounding between -1000 and 400 captures both effectively.
                ct_resampled = sitk.Clamp(ct_resampled, lowerBound=-1000.0, upperBound=400.0)
                
                # Save
                sitk.WriteImage(ct_resampled, os.path.join(patient_out_dir, "ct_resampled.nii.gz"))
            else:
                logging.warning(f"No CT path provided for {patient_id}")

            # Process PET
            if pet_series_path:
                print(f"Processing PET for {patient_id}...")
                pet_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(pet_series_path)
                pet_img = sitk.ReadImage(pet_names)
                
                # Check for RGB/Vector
                if pet_img.GetNumberOfComponentsPerPixel() > 1:
                    logging.warning(f"PET {patient_id} is Vector/RGB. Converting to Grayscale.")
                    pet_img = sitk.VectorIndexSelectionCast(pet_img, 0)
                
                # Convert to SUV (Approximate if tags missing, but SimpleITK reads raw values)
                # Ideally we need specialized SUV conversion, but for now we assume raw PET activity or corrected values
                
                # Add Clinical SUV range constraint (0-15)
                pet_resampled = self.resample_image(pet_img, self.target_spacing)
                pet_resampled = sitk.Clamp(pet_resampled, lowerBound=0.0, upperBound=15.0)
                
                # Save
                sitk.WriteImage(pet_resampled, os.path.join(patient_out_dir, "pet_resampled.nii.gz"))
            else:
                logging.warning(f"No PET path provided for {patient_id}")
                
            logging.info(f"Successfully processed {patient_id}")
            
        except Exception as e:
            logging.error(f"Error processing {patient_id}: {e}")
            print(f"Error: {e}")

    def run_batch(self, metadata_csv):
        """Runs processing for all patients in metadata CSV."""
        if not os.path.exists(metadata_csv):
            print("Metadata CSV not found. Run dataset_explorer first.")
            return

        df = pd.read_csv(metadata_csv)
        
        # We need to reconstruct the pairs from the metadata
        # Group by PatientID
        unique_patients = df['PatientID'].unique()
        
        for pid in tqdm(unique_patients, desc="Preprocessing Patients"):
            patient_rows = df[df['PatientID'] == pid]
            
            ct_row = patient_rows[patient_rows['Modality'] == 'CT']
            pet_row = patient_rows[patient_rows['Modality'].isin(['PT', 'NM'])]
            
            ct_path = ct_row.iloc[0]['SeriesPath'] if not ct_row.empty else None
            pet_path = pet_row.iloc[0]['SeriesPath'] if not pet_row.empty else None
            
            if ct_path and pet_path:
                self.process_patient(pid, ct_path, pet_path)

if __name__ == "__main__":
    agent = DICOMPreprocessingAgent()
    agent.run_batch("metadata.csv")
