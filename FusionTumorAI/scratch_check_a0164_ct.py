import os
import SimpleITK as sitk
import pyvista as pv

pid = "Lung_Dx-A0164"
p_dir = f"c:\\Users\\ks181\\Music\\lung cancer model\\FusionTumorAI\\data\\processed\\{pid}"
ct_file = os.path.join(p_dir, "ct_cropped.nii.gz")

ct_sitk = sitk.ReadImage(ct_file)
ct_arr = sitk.GetArrayFromImage(ct_sitk)

print(f"CT Array max: {ct_arr.max()}, min: {ct_arr.min()}")
