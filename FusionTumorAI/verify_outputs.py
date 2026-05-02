import os, json

processed = "data/processed"
patients = sorted([d for d in os.listdir(processed) if os.path.isdir(os.path.join(processed, d))])
print(f"Total patients: {len(patients)}")
print()
header = f"{'Patient':<20} {'HTML':>5} {'PNG':>5} {'GIF':>5} {'GLTF':>5} {'JSON':>5} {'PDF':>5}"
print(header)
print("-" * 65)
for p in patients:
    pp = os.path.join(processed, p)
    html = "Y" if os.path.exists(os.path.join(pp, "3d_viewer.html")) else "N"
    png = "Y" if os.path.exists(os.path.join(pp, "3d_render.png")) else "N"
    gif = "Y" if os.path.exists(os.path.join(pp, "3d_rotation.gif")) else "N"
    gltf = "Y" if os.path.exists(os.path.join(pp, "tumor_mesh.gltf")) else "N"
    rj = "Y" if os.path.exists(os.path.join(pp, "radiomics.json")) else "N"
    pdf = "Y" if os.path.exists(os.path.join("reports", f"{p}_report.pdf")) else "N"
    print(f"{p:<20} {html:>5} {png:>5} {gif:>5} {gltf:>5} {rj:>5} {pdf:>5}")
