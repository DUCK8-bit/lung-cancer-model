import os
import json
import SimpleITK as sitk
import logging
from tqdm import tqdm

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/registration.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class RegistrationAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.processed_dir = self.config["processed_dir"]

    def register_pet_to_ct(self, fixed_image, moving_image):
        """Performs rigid registration between CT (fixed) and PET (moving)."""
        
        # Initial Alignment
        initial_transform = sitk.CenteredTransformInitializer(
            fixed_image, 
            moving_image, 
            sitk.Euler3DTransform(), 
            sitk.CenteredTransformInitializerFilter.GEOMETRY
        )

        # Registration Method
        registration_method = sitk.ImageRegistrationMethod()
        
        # Similarity Metric
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.01)

        # Optimizer
        registration_method.SetOptimizerAsGradientDescent(
            learningRate=1.0, 
            numberOfIterations=100, 
            convergenceMinimumValue=1e-6, 
            convergenceWindowSize=10
        )
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # Setup
        registration_method.SetInitialTransform(initial_transform, inPlace=False)
        registration_method.SetInterpolator(sitk.sitkLinear)

        # Execute
        final_transform = registration_method.Execute(fixed_image, moving_image)
        
        print(f"Final Metric Value: {registration_method.GetMetricValue()}")
        print(f"Optimizer stop condition: {registration_method.GetOptimizerStopConditionDescription()}")
        
        return final_transform

    def process_patient(self, patient_id):
        patient_path = os.path.join(self.processed_dir, patient_id)
        ct_path = os.path.join(patient_path, "ct_resampled.nii.gz")
        pet_path = os.path.join(patient_path, "pet_resampled.nii.gz")
        
        if not os.path.exists(ct_path) or not os.path.exists(pet_path):
            logging.warning(f"Missing data for registration in {patient_id}")
            return

        aligned_path = os.path.join(patient_path, "pet_aligned.nii.gz")
        transform_path = os.path.join(patient_path, "registration_transform.tfm")

        if os.path.exists(aligned_path) and os.path.exists(transform_path):
            print(f"Skipping {patient_id} - already registered.")
            return

        try:
            print(f"Registering PET to CT for {patient_id}...")
            ct_img = sitk.ReadImage(ct_path, sitk.sitkFloat32)
            pet_img = sitk.ReadImage(pet_path, sitk.sitkFloat32)
            
            transform = self.register_pet_to_ct(ct_img, pet_img)
            
            # Resample PET using the transform to match CT space perfectly
            resampled_pet = sitk.Resample(
                pet_img, 
                ct_img, 
                transform, 
                sitk.sitkLinear, 
                0.0, 
                pet_img.GetPixelID()
            )
            
            # Save aligned PET
            sitk.WriteImage(resampled_pet, os.path.join(patient_path, "pet_aligned.nii.gz"))
            
            # Save transform
            sitk.WriteTransform(transform, os.path.join(patient_path, "registration_transform.tfm"))
            
            logging.info(f"Registration successful for {patient_id}")
            
        except Exception as e:
            logging.error(f"Registration failed for {patient_id}: {e}")
            print(f"Error: {e}")

    def run_batch(self):
        """Runs registration for all processed patients."""
        if not os.path.exists(self.processed_dir):
            print("No processed data found.")
            return

        patients = [d for d in os.listdir(self.processed_dir) if os.path.isdir(os.path.join(self.processed_dir, d))]
        
        for pid in tqdm(patients, desc="Registering Volumes"):
            self.process_patient(pid)

if __name__ == "__main__":
    agent = RegistrationAgent()
    agent.run_batch()
