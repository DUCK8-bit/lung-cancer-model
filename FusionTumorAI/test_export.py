import pyvista as pv
import numpy as np
import os

def test_export():
    plotter = pv.Plotter(off_screen=True)
    
    # Create a dummy sphere (representing tumor)
    sphere = pv.Sphere(radius=10)
    sphere["Scalars"] = sphere.points[:, 2] # Add dummy scalar data
    
    plotter.add_mesh(sphere, scalars="Scalars", cmap="hot")
    
    # Add measurements/scale just like a map
    plotter.show_grid(xlabel='X (mm)', ylabel='Y (mm)', zlabel='Z (mm)')
    
    plotter.view_isometric()
    
    # Try exporting to HTML
    try:
        plotter.export_html("test_interactive.html")
        print("Success: HTML export worked.")
    except Exception as e:
        print(f"Failed to export HTML: {e}")

    # Try exporting to GLTF
    try:
        plotter.export_gltf("test_interactive.gltf")
        print("Success: GLTF export worked.")
    except Exception as e:
        print(f"Failed to export GLTF: {e}")

if __name__ == "__main__":
    test_export()
