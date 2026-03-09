import sys
import traceback

try:
    import torch
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA NOT AVAILABLE")
except Exception:
    with open("gpu_error.log", "w") as f:
        traceback.print_exc(file=f)
    print("Import Failed. Check gpu_error.log")
