import os
import SimpleITK as sitk
import pyvista as pv

pid = "Lung_Dx-A0166"
p_dir = f"c:\\Users\\ks181\\Music\\lung cancer model\\FusionTumorAI\\data\\processed\\{pid}"
ct_file = os.path.join(p_dir, "ct_cropped.nii.gz")

ct_sitk = sitk.ReadImage(ct_file)
ct_arr = sitk.GetArrayFromImage(ct_sitk)

print(f"CT Array max: {ct_arr.max()}, min: {ct_arr.min()}")

grid = pv.wrap(ct_arr)
print("Wrapped grid.")

lung_vol = grid.threshold([-1000, -400])
print(f"Lung Vol points: {lung_vol.n_points}, cells: {lung_vol.n_cells}")

if lung_vol.n_points > 0:
    lung_mesh = lung_vol.extract_surface()
    print(f"Extracted surface points: {lung_mesh.n_points}, cells: {lung_mesh.n_cells}")
    
    lung_mesh_gltf = lung_mesh.triangulate().decimate(0.95)
    print(f"Decimated surface points: {lung_mesh_gltf.n_points}, cells: {lung_mesh_gltf.n_cells}")
else:
    print("No lung volume extracted.")
