import os
import pydicom
import pandas as pd
import json
import logging
from tqdm import tqdm
from datetime import datetime

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/dataset_explorer.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class DatasetExplorerAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.data_root = self.config["data_root"]
        self.metadata = []
        self.stats = {}
        
    def scan_dataset(self):
        """Scans the dataset directory for DICOM files."""
        # Ensure imports are available
        import pandas as pd
        import os

        print(f"Scanning dataset in {self.data_root}...")
        logging.info(f"Scanning dataset in {self.data_root}...")
        
        if not os.path.exists(self.data_root):
            logging.error(f"Data root not found: {self.data_root}")
            print(f"Error: Data root not found: {self.data_root}")
            return pd.DataFrame()

        # Depending on dataset structure (e.g., manifest/Lung-PET-CT-Dx)
        # We walk to find series
        
        for root, dirs, files in os.walk(self.data_root):
            dicom_files = [f for f in files if f.endswith('.dcm')]
            if not dicom_files:
                continue

            # Heuristic: Check Modality of first file
            try:
                dcm = pydicom.dcmread(os.path.join(root, dicom_files[0]), stop_before_pixels=True)
                pid = dcm.PatientID
                modality = dcm.Modality
                study_uid = dcm.StudyInstanceUID
                series_uid = dcm.SeriesInstanceUID
                
                self.metadata.append({
                    "PatientID": pid,
                    "Modality": modality,
                    "SeriesPath": root,
                    "NumSlices": len(dicom_files),
                    "StudyUID": study_uid,
                    "SeriesUID": series_uid
                })
            except Exception as e:
                logging.warning(f"Failed to read DICOM in {root}: {e}")

        df = pd.DataFrame(self.metadata)
        return df

    def validate_pairing(self, df):
        """Checks if each patient has both PET and CT data."""
        import pandas as pd
        
        if df.empty:
            print("No data found.")
            return df
            
        # Group by PatientID and Check for CT and PT
        validation_results = []
        unique_patients = df['PatientID'].unique()
        
        for pid in unique_patients:
            subset = df[df['PatientID'] == pid]
            has_ct = 'CT' in subset['Modality'].values
            has_pet = 'PT' in subset['Modality'].values or 'NM' in subset['Modality'].values # PT is standard, ensuring NM not missed
            
            validation_results.append({
                "PatientID": pid,
                "HasCT": has_ct,
                "HasPET": has_pet,
                "ValidPair": has_ct and has_pet
            })
            
        val_df = pd.DataFrame(validation_results)
        valid_count = val_df['ValidPair'].sum()
        total_count = len(val_df)
        
        print(f"Dataset Scan Complete. Found {valid_count}/{total_count} valid PET/CT pairs.")
        logging.info(f"Dataset Scan Complete. Found {valid_count}/{total_count} valid PET/CT pairs.")
        
        self.stats['total_patients'] = total_count
        self.stats['valid_pairs'] = int(valid_count) # generic int for json serialization
        
        return val_df

    def generate_report(self, val_df, full_df):
        """Generates a summary report and saves metadata."""
        os.makedirs(self.config["reports_dir"], exist_ok=True)
        report_path = os.path.join(self.config["reports_dir"], "data_quality_report.txt")
        
        with open(report_path, "w") as f:
            f.write("FusionTumorAI - Data Quality Report\n")
            f.write("===================================\n")
            f.write(f"Generated on: {datetime.now()}\n\n")
            f.write(f"Total Patients Scanned: {self.stats.get('total_patients', 0)}\n")
            f.write(f"Valid PET/CT Pairs:     {self.stats.get('valid_pairs', 0)}\n")
            
            if self.stats.get('valid_pairs', 0) == 0:
                f.write("CRITICAL WARNING: No valid PET-CT pairs found! Check dataset structure.\n")
        
        full_df.to_csv("metadata.csv", index=False)
        val_df.to_csv("validation_results.csv", index=False)
        with open("dataset_statistics.json", "w") as f:
            json.dump(self.stats, f, indent=4)
            
        print(f"Report validated. Metadata saved to metadata.csv")
        logging.info("Report generated and metadata saved.")

if __name__ == "__main__":
    agent = DatasetExplorerAgent()
    df = agent.scan_dataset()
    val_df = agent.validate_pairing(df)
    agent.generate_report(val_df, df)
