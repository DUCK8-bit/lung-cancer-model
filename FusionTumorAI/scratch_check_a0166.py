import os
import SimpleITK as sitk
import pyvista as pv

pid = "Lung_Dx-A0166"
p_dir = f"c:\\Users\\ks181\\Music\\lung cancer model\\FusionTumorAI\\data\\processed\\{pid}"
lung_file = os.path.join(p_dir, "lung_mask.nii.gz")
tumor_file = os.path.join(p_dir, "tumor_mask.nii.gz")

print(f"Checking {pid}...")
if os.path.exists(lung_file):
    lung_sitk = sitk.ReadImage(lung_file)
    lung_arr = sitk.GetArrayFromImage(lung_sitk)
    print(f"Lung Mask shape: {lung_arr.shape}, max: {lung_arr.max()}, sum: {lung_arr.sum()}")
    lung_vol = pv.wrap(lung_sitk)
    try:
        lung_mesh = lung_vol.marching_cubes(level=0.5)
        print(f"Lung Mesh points: {lung_mesh.n_points}, cells: {lung_mesh.n_cells}")
    except Exception as e:
        print(f"Lung marching cubes error: {e}")
else:
    print("No lung_mask.nii.gz")

if os.path.exists(tumor_file):
    tumor_sitk = sitk.ReadImage(tumor_file)
    tumor_arr = sitk.GetArrayFromImage(tumor_sitk)
    print(f"Tumor Mask shape: {tumor_arr.shape}, max: {tumor_arr.max()}, sum: {tumor_arr.sum()}")
    tumor_vol = pv.wrap(tumor_sitk)
    try:
        tumor_mesh = tumor_vol.marching_cubes(level=0.5)
        print(f"Tumor Mesh points: {tumor_mesh.n_points}, cells: {tumor_mesh.n_cells}")
    except Exception as e:
        print(f"Tumor marching cubes error: {e}")
