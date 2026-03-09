import ctypes
import os

dll_path = r"c:\Users\ks181\Music\lung cancer model\LungFusion-AI\venv\lib\site-packages\torch\lib\cufft64_11.dll"
try:
    ctypes.cdll.LoadLibrary(dll_path)
    print("Successfully loaded DLL")
except Exception as e:
    print(f"Failed to load DLL: {e}")
