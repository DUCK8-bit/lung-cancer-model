
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import torch
import SimpleITK as sitk
try:
    import radiomics
    HAS_PYRADIOMICS = True
except ImportError:
    HAS_PYRADIOMICS = False

print(f"Python Version: {sys.version}")
print("-" * 30)

print(f"PyTorch Version: {torch.__version__}")
if torch.cuda.is_available():
    print(f"CUDA Available: YES (Device: {torch.cuda.get_device_name(0)})")
else:
    print("CUDA Available: NO (Using CPU)")

print("-" * 30)

print(f"SimpleITK Version: {sitk.Version()}")

if HAS_PYRADIOMICS:
    print(f"PyRadiomics Version: {radiomics.__version__}")
else:
    print("PyRadiomics: NOT INSTALLED")

print("-" * 30)
print("Environment Verification Complete.")
