import os
import sys
import json
import logging
import numpy as np
import SimpleITK as sitk

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.inference import InferenceAgent
from agents.visualization import VisualizationAgent
from agents.report_generator import ReportGenerationAgent

# Setup logging
logging.basicConfig(level=logging.INFO)

class DiagnosticService:
    def __init__(self):
        self.inference_agent = InferenceAgent()
        self.viz_agent = VisualizationAgent()
        self.report_agent = ReportGenerationAgent()
        self.processed_dir = self.inference_agent.processed_dir

    def calculate_infection_risk(self, suv_max, mean_hu):
        """
        Infection Guard Logic:
        High SUV (> 2.5) but Ground Glass Opacity (-500 < HU < -200) often indicates infection/inflammatory process (e.g., TB, Pneumonia) rather than solid tumor.
        """
        suv_threshold = 2.5
        hu_min = -500
        hu_max = -200
        
        status = "Normal"
        description = "No anomalies detected."
        
        if suv_max > suv_threshold:
            if hu_min < mean_hu < hu_max:
                status = "Risk: Infection/TB"
                description = f"High Metabolic Activity (SUV {suv_max:.2f}) with Ground Glass Opacity (HU {mean_hu:.0f}). Potential inflammatory process."
            else:
                status = "Risk: Malignancy"
                description = f"High Metabolic Activity (SUV {suv_max:.2f}) with Solid Tissue Density (HU {mean_hu:.0f}). Likely malignant."
        else:
            status = "Low Risk"
            description = f"Low Metabolic Activity (SUV {suv_max:.2f}). Likely benign."
            
        return status, description

    def get_patient_metrics(self, patient_id):
        """
        Extracts metrics from the processed data for a given patient.
        """
        patient_path = os.path.join(self.processed_dir, patient_id)
        prediction_path = os.path.join(patient_path, "prediction.nii.gz")
        ct_path = os.path.join(patient_path, "ct_cropped.nii.gz")
        pet_path = os.path.join(patient_path, "pet_cropped.nii.gz")
        
        if not os.path.exists(prediction_path):
            return None

        # Load images
        mask_img = sitk.ReadImage(prediction_path)
        ct_img = sitk.ReadImage(ct_path)
        pet_img = sitk.ReadImage(pet_path)
        
        mask_arr = sitk.GetArrayFromImage(mask_img)
        ct_arr = sitk.GetArrayFromImage(ct_img)
        pet_arr = sitk.GetArrayFromImage(pet_img)
        
        # Calculate stats within the mask
        if np.sum(mask_arr) > 0:
            tumor_mask = mask_arr > 0
            suv_max = float(np.max(pet_arr[tumor_mask]))
            mean_hu = float(np.mean(ct_arr[tumor_mask]))
            volume_cm3 = float(np.sum(mask_arr) * np.prod(mask_img.GetSpacing()) / 1000.0)
        else:
             # Fallback if no tumor detected
             suv_max = 0.0
             mean_hu = -1000.0
             volume_cm3 = 0.0
             
        risk_status, risk_desc = self.calculate_infection_risk(suv_max, mean_hu)
        
        return {
            "patient_id": patient_id,
            "tumor_volume_cm3": round(volume_cm3, 2),
            "max_suv": round(suv_max, 2),
            "mean_hu": round(mean_hu, 2),
            "risk_assessment": {
                "status": risk_status,
                "description": risk_desc
            }
        }

    def run_prediction(self, patient_id):
        # Trigger Inference
        self.inference_agent.run_inference(patient_id)
        # Return Metrics
        return self.get_patient_metrics(patient_id)

    def generate_visualization(self, patient_id):
        # Trigger Visualization
        self.viz_agent.create_3d_snapshot(patient_id)
        # Return path to image
        img_path = os.path.join(self.processed_dir, patient_id, "3d_render.png")
        if os.path.exists(img_path):
            return img_path
        return None

    def get_viewer(self, patient_id):
        html_path = os.path.join(self.processed_dir, patient_id, "3d_viewer.html")
        if os.path.exists(html_path):
            return html_path
        return None

    def get_gltf(self, patient_id):
        gltf_path = os.path.join(self.processed_dir, patient_id, "tumor_mesh.gltf")
        if os.path.exists(gltf_path):
            return gltf_path
        return None

    def generate_report(self, patient_id):
        # Trigger Report
        self.report_agent.generate_report(patient_id)
        # Return path to PDF
        pdf_path = os.path.join(self.report_agent.reports_dir, f"{patient_id}_report.pdf")
        if os.path.exists(pdf_path):
            return pdf_path
        return None
