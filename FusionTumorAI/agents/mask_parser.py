import os
import glob
import xml.etree.ElementTree as ET
import numpy as np
import SimpleITK as sitk
import pydicom
import json
from tqdm import tqdm
import logging

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/mask_parser.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class MaskParser:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.data_root = self.config["data_root"]
        self.processed_dir = self.config["processed_dir"]
        
        # Adjust path to Annotation folder based on observed structure
        # Validating two potential locations
        self.annot_root = os.path.join(self.data_root, "Annotation", "Annotation")
        if not os.path.exists(self.annot_root):
             self.annot_root = os.path.join(self.data_root, "Annotation")

    def parse_xml(self, xml_file):
        """Parses XML to extract ROI coordinates."""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Expected structure: <unblindedReadNodule> ... <roi> ... <edgeMap> ... <xCoord>, <yCoord>
        # Or similar. Adjusting to Lung-PET-CT-Dx XML format.
        
        rois = []
        for reading_session in root.findall("readingSession"):
            for nodule in reading_session.findall("unblindedReadNodule"):
                # Check if malignant? For now, take all nodules.
                # Characteristics are in <characteristics>
                
                for roi in nodule.findall("roi"):
                    z_pos = float(roi.find("imageZposition").text)
                    sop_uid = roi.find("imageSOP_UID").text
                    
                    edge_points = []
                    for edge in roi.findall("edgeMap"):
                        x = int(edge.find("xCoord").text)
                        y = int(edge.find("yCoord").text)
                        edge_points.append((x, y))
                    
                    rois.append({
                        "z": z_pos,
                        "sop_uid": sop_uid,
                        "points": edge_points
                    })
        return rois

    def create_mask_from_rois(self, rois, ct_series_path, output_path):
        """Creates a 3D NIfTI mask from parsed ROIs matching CT series."""
        
        # Load CT DICOMs to get spatial reference
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(ct_series_path)
        reader.SetFileNames(dicom_names)
        ct_img = reader.Execute()
        
        if not rois:
            logging.warning(f"No ROIs found for {ct_series_path}")
            # Create empty mask
            mask_img = sitk.Image(ct_img.GetSize(), sitk.sitkUInt8)
            mask_img.CopyInformation(ct_img)
            sitk.WriteImage(mask_img, output_path)
            return

        # Create empty numpy array
        size = ct_img.GetSize()
        mask_arr = np.zeros((size[2], size[1], size[0]), dtype=np.uint8) # Z, Y, X
        
        # Map SOPInstanceUID to Z-index
        # We need to read dataset to map SOP UID -> Slice Index
        # This is slow if we read all files. 
        # Optimization: Trust Image Position Patient Z if available or sort by location
        
        # Let's map Z-position to slice index directly if uniform spacing
        # Origin Z + Index * Spacing Z = Z-pos
        origin = ct_img.GetOrigin()
        spacing = ct_img.GetSpacing()
        
        for roi in rois:
            z_coord = roi['z']
            points = roi['points']
            
            # Find closest slice index
            z_idx = int(round((z_coord - origin[2]) / spacing[2]))
            
            if 0 <= z_idx < size[2]:
                # Draw polygon on slice
                from skimage.draw import polygon
                
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                
                rr, cc = polygon(ys, xs, shape=(size[1], size[0]))
                mask_arr[z_idx, rr, cc] = 1
                
        # Convert to Sitk
        mask_img = sitk.GetImageFromArray(mask_arr)
        mask_img.CopyInformation(ct_img)
        
        sitk.WriteImage(mask_img, output_path)

    def process_patient(self, annot_pid, target_pid):
        # Find XML file using Annotation ID
        patient_annot_dir = os.path.join(self.annot_root, annot_pid)
        if not os.path.exists(patient_annot_dir):
            logging.warning(f"No annotation dir for {annot_pid}")
            print(f"No annotation dir for {annot_pid} at {patient_annot_dir}")
            return

        # Find XML (usually named with UID or number)
        xml_files = glob.glob(os.path.join(patient_annot_dir, "*.xml"))
        if not xml_files:
            logging.warning(f"No XML found in {patient_annot_dir}")
            print(f"No XML found in {patient_annot_dir}")
            return
            
        # Parse all XMLs (could be multiple reads)
        all_rois = []
        for xml_f in xml_files:
            try:
                rois = self.parse_xml(xml_f)
                all_rois.extend(rois)
            except Exception as e:
                logging.error(f"Error parsing {xml_f}: {e}")
                print(f"Error parsing {xml_f}: {e}")

        # Find CT Series for target_pid regardless of ROIs
        # Look in likely locations
        possible_paths = [
            os.path.join(self.data_root, "Lung-PET-CT-Dx", target_pid),
            os.path.join(self.data_root, "Lung-PET-CT-Dx", "Lung-PET-CT-Dx", target_pid),
            os.path.join(self.data_root, target_pid),
            os.path.join(self.data_root, "Lung-PET-CT-Dx", annot_pid)
        ]

        ct_series_path = None
        for path in possible_paths:
            if not os.path.exists(path): continue
            
            for root, _, files in os.walk(path):
                if any(f.endswith('.dcm') for f in files):
                     try:
                         # Quick check first file
                         dcm = pydicom.dcmread(os.path.join(root, files[0]), stop_before_pixels=True)
                         if dcm.Modality == 'CT':
                             ct_series_path = root
                             break
                     except: pass
            if ct_series_path: break
        
        if not all_rois:
            logging.warning(f"No ROIs extracted from {len(xml_files)} XML files for {annot_pid}")
            print(f"No ROIs extracted for {annot_pid}. Proceeding to create blank mask if CT found.")
        
        if ct_series_path:
            # Output to target_pid folder in processed
            output_mask_path = os.path.join(self.processed_dir, target_pid, "mask_original.nii.gz")
            os.makedirs(os.path.dirname(output_mask_path), exist_ok=True)
            self.create_mask_from_rois(all_rois, ct_series_path, output_mask_path)
            logging.info(f"Created mask for {target_pid}")
            print(f"Created mask for {target_pid}")
        else:
            logging.warning(f"CT series not found for {target_pid}")
            print(f"CT series not found for {target_pid} in {possible_paths}")

    def run_batch(self):
        # Get list of processed patients to filter work
        if os.path.exists(self.processed_dir):
            processed_patients = set(os.listdir(self.processed_dir))
        else:
            processed_patients = set()

        annot_patients = [d for d in os.listdir(self.annot_root) if os.path.isdir(os.path.join(self.annot_root, d))]
        # annot_patients = ["A0164"] # DEBUG ONLY
        
        for pid in tqdm(annot_patients, desc="Parsing Masks"):
            # Map Annotation ID (Axxxx) to Processed ID (Lung_Dx-Axxxx)
            # Try direct match or prefix match
            target_pid = pid
            print(f"Processing PID: {pid}")
            if pid not in processed_patients:
                if f"Lung_Dx-{pid}" in processed_patients:
                    target_pid = f"Lung_Dx-{pid}"
                    print(f"Mapped {pid} to {target_pid}")
                else:
                    # Skip if patient not in processed list (we only need masks for active subset)
                    print(f"Skipping {pid} - not in processed_patients: {list(processed_patients)[:5]}...")
                    continue
            
            print(f"Calling process_patient({pid}, {target_pid})")
            self.process_patient(pid, target_pid)

if __name__ == "__main__":
    parser = MaskParser()
    parser.run_batch()
