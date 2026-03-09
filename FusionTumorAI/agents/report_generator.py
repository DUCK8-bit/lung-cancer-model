import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import logging
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from tqdm import tqdm

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/reporting.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class ReportGenerationAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.processed_dir = self.config["processed_dir"]
        self.reports_dir = self.config["reports_dir"]
        self.models_dir = self.config["models_dir"]
        os.makedirs(self.reports_dir, exist_ok=True)

    def generate_report(self, patient_id):
        patient_path = os.path.join(self.processed_dir, patient_id)
        
        # Load Data
        img_path = os.path.join(patient_path, "3d_render.png")
        gradcam_path = os.path.join(patient_path, "gradcam.nii.gz") # Can't put nii in PDF. Need PNG.
        # ExplainabilityAgent should have saved a PNG slice for report ideally. 
        # For now, we use the 3D render.
        
        # Radiomics
        features_csv = os.path.join(self.models_dir, "radiomics_features.csv")
        metrics = {}
        
        # Try loading per-patient JSON first (more robust)
        json_path = os.path.join(patient_path, "radiomics.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metrics = json.load(f)
        elif os.path.exists(features_csv):
            df = pd.read_csv(features_csv)
            row = df[df['PatientID'] == patient_id]
            if not row.empty:
                metrics = row.iloc[0].to_dict()

        # Dynamic Diagnosis Prediction (Infection Guard)
        suv_max = metrics.get('PET_original_firstorder_Maximum', 0.0)
        mean_hu = metrics.get('CT_original_firstorder_Mean', -1000.0)
        
        # Ensure types
        if isinstance(suv_max, str): suv_max = float(suv_max)
        if isinstance(mean_hu, str): mean_hu = float(mean_hu)
        
        diagnosis = "Benign/Indeterminate"
        if suv_max > 2.5:
            if mean_hu > -400:
                diagnosis = "Malignant Suspicion (Adenocarcinoma)"
            elif -900 <= mean_hu <= -500:
                diagnosis = "Potential Infection (TB/Pneumonia)"
            else:
                 diagnosis = "Suspicious (Uncertain Etiology)"
        
        # RECIST
        recist_dia = metrics.get('CT_original_shape_Maximum3DDiameter', 'N/A')
        if isinstance(recist_dia, (int, float)):
            recist_str = f"{recist_dia:.2f} mm"
        else:
            recist_str = "N/A"

        # Build PDF (Restored)
        doc_path = os.path.join(self.reports_dir, f"{patient_id}_report.pdf")
        doc = SimpleDocTemplate(doc_path, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph(f"FusionTumorAI Analysis Report - {patient_id}", styles['Title']))
        story.append(Spacer(1, 12))

        # Summary Table
        data = [
            ["Metric", "Value"],
            ["Patient ID", patient_id],
            ["Diagnosis Prediction", diagnosis],
            ["Tumor Volume (cm³)", f"{metrics.get('Tumor_Volume_cm3', 'N/A'):.2f}" if isinstance(metrics.get('Tumor_Volume_cm3'), (int, float)) else "N/A"],
            ["RECIST Diameter (Max 3D)", recist_str],
            ["SUV Max", f"{suv_max:.2f}"],
            ["Mean HU", f"{mean_hu:.2f}"],
            ["Sphericity", f"{metrics.get('CT_original_shape_Sphericity', 'N/A'):.2f}" if isinstance(metrics.get('CT_original_shape_Sphericity'), (int, float)) else "N/A"],
            ["Entropy", f"{metrics.get('CT_original_firstorder_Entropy', 'N/A'):.2f}" if isinstance(metrics.get('CT_original_firstorder_Entropy'), (int, float)) else "N/A"]
        ]
        
        t = Table(data)
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(t)
        story.append(Spacer(1, 24))
        
        # 3D Visualization
        if os.path.exists(img_path):
            story.append(Paragraph("3D Tumor Visualization", styles['Heading2']))
            im = Image(img_path, width=400, height=300)
            story.append(im)
            
        story.append(Spacer(1, 12))
        story.append(Paragraph("This report was generated automatically by FusionTumorAI.", styles['Italic']))
        
        doc.build(story)
        logging.info(f"Report generated for {patient_id}")

    def run_batch(self):
        patients = [d for d in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, d))]
        for pid in tqdm(patients, desc="Generating Reports"):
            self.generate_report(pid)

if __name__ == "__main__":
    agent = ReportGenerationAgent()
    agent.run_batch()
