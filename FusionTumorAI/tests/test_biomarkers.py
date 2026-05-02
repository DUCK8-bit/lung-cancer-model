import os
import json
import numpy as np
import SimpleITK as sitk
import pytest
from agents.radiomics import RadiomicsAgent

@pytest.fixture
def synthetic_patient_data(tmp_path):
    patient_id = "Test_Patient_001"
    processed_dir = tmp_path / "processed"
    patient_dir = processed_dir / patient_id
    patient_dir.mkdir(parents=True)
    
    # Create simple synthetic CT (10x10x10) - Random values between -1000 and 200 HI
    ct_arr = np.random.uniform(-1000, 200, size=(10, 10, 10)).astype(np.float32)
    ct_img = sitk.GetImageFromArray(ct_arr)
    ct_img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(ct_img, str(patient_dir / "ct_cropped.nii.gz"))
    
    # Create synthetic mask (center 4x4x4 cube)
    mask_arr = np.zeros((10, 10, 10), dtype=np.uint8)
    mask_arr[3:7, 3:7, 3:7] = 1
    mask_img = sitk.GetImageFromArray(mask_arr)
    mask_img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(mask_img, str(patient_dir / "prediction.nii.gz"))
    
    # Create synthetic PET - Max value 8.5 inside the mask
    pet_arr = np.zeros((10, 10, 10), dtype=np.float32)
    pet_arr[4, 4, 4] = 8.5
    pet_img = sitk.GetImageFromArray(pet_arr)
    pet_img.SetSpacing((1.0, 1.0, 1.0))
    sitk.WriteImage(pet_img, str(patient_dir / "pet_cropped.nii.gz"))
    
    # Create dummy config
    config_path = tmp_path / "config.json"
    with open(config_path, "w") as f:
        json.dump({"processed_dir": str(processed_dir), "models_dir": str(tmp_path)}, f)
        
    return str(config_path), patient_id

def test_biomarker_extraction(synthetic_patient_data):
    config_path, patient_id = synthetic_patient_data
    
    agent = RadiomicsAgent(config_path=config_path)
    # Disable pyradiomics for speed
    agent.extractor = None 
    
    features = agent.extract_features(patient_id)
    
    assert features is not None, "Extraction failed"
    
    # Verify SUV Max (Should be 8.5 based on synthetic PET)
    suv_max = features.get("PET_original_firstorder_Maximum", None)
    assert suv_max is not None
    assert isinstance(suv_max, float)
    assert 8.4 < suv_max < 8.6
    
    # Verify Entropy (Should be a float, not 4.5 placeholder)
    entropy = features.get("CT_original_firstorder_Entropy", None)
    assert entropy is not None
    assert isinstance(entropy, float)
    assert entropy != 4.5  # Ensure the placeholder is gone
    assert entropy > 0.0   # Real entropy is positive
    
    # Verify Sphericity (Cube sphericity is ~0.806, well within [0,1])
    sphericity = features.get("CT_original_shape_Sphericity", None)
    assert sphericity is not None
    assert isinstance(sphericity, float)
    assert 0.0 <= sphericity <= 1.0
    assert sphericity > 0.5 # A 4x4x4 cube should be relatively spherical > 0.5
