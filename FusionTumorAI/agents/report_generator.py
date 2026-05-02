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

        # Threshold flags for the table
        suv_flag = " (Hypermetabolic)" if suv_max > 2.5 else ""
        
        sph_val = metrics.get('CT_original_shape_Sphericity', 'N/A')
        sph_flag = ""
        if isinstance(sph_val, (int, float)):
            sph_str = f"{sph_val:.2f}"
            sph_flag = " (Irregular)" if sph_val < 0.75 else " (Regular)"
        else:
            sph_str = "N/A"
            
        ent_val = metrics.get('CT_original_firstorder_Entropy', 'N/A')
        ent_flag = ""
        if isinstance(ent_val, (int, float)):
            ent_str = f"{ent_val:.2f}"
            ent_flag = " (Heterogeneous)" if ent_val > 4.5 else " (Homogeneous)"
        else:
            ent_str = "N/A"
        
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
            ["SUV Max", f"{suv_max:.2f}{suv_flag}"],
            ["Mean HU", f"{mean_hu:.2f}"],
            ["Sphericity", f"{sph_str}{sph_flag}"],
            ["Entropy", f"{ent_str}{ent_flag}"]
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
        
        # Clinical Interpretation Section
        story.append(Paragraph("Clinical Biomarker Interpretation", styles['Heading3']))
        
        interp_text = []
        # SUV Interpretation
        if suv_max > 2.5:
            interp_text.append(f"<b>Metabolic (SUV Max):</b> A value of {suv_max:.2f} exceeds the 2.5 threshold, strongly indicating a hypermetabolic state commonly associated with malignancy or active infection.")
        else:
            interp_text.append(f"<b>Metabolic (SUV Max):</b> A value of {suv_max:.2f} is below the 2.5 threshold, indicating lower metabolic activity.")
            
        # Sphericity Interpretation
        if isinstance(sph_val, (int, float)):
            if sph_val < 0.75:
                interp_text.append(f"<b>Shape (Sphericity):</b> A value of {sph_val:.2f} indicates irregular tumor borders, which is a key predictor of invasiveness and aggressive cell growth.")
            else:
                interp_text.append(f"<b>Shape (Sphericity):</b> A value of {sph_val:.2f} indicates relatively regular/rounded borders.")
                
        # Entropy Interpretation
        if isinstance(ent_val, (int, float)):
            if ent_val > 4.5:
                interp_text.append(f"<b>Texture (Entropy):</b> A value of {ent_val:.2f} indicates high internal heterogeneity ('messy' texture), which literature correlates with poor prognosis.")
            else:
                interp_text.append(f"<b>Texture (Entropy):</b> A value of {ent_val:.2f} indicates relatively homogeneous internal texture.")

        for text in interp_text:
            story.append(Paragraph(text, styles['Normal']))
            story.append(Spacer(1, 6))
            
        story.append(Spacer(1, 18))
        
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
