import os
import sys
from tqdm import tqdm
from agents.visualization import VisualizationAgent

def main():
    agent = VisualizationAgent()
    
    # Priority process A0166 so the user can view it immediately
    target_pid = "Lung_Dx-A0166"
    print(f"Priority rebuilding 3D meshes for {target_pid}...")
    try:
        agent.create_3d_snapshot(target_pid)
        print(f"Successfully rebuilt {target_pid}!")
    except Exception as e:
        print(f"Error on {target_pid}: {e}")
        
    # Then rebuild the rest that need the new GLTF format
    patients = [d for d in os.listdir(agent.processed_dir) if os.path.isdir(os.path.join(agent.processed_dir, d))]
    
    for pid in tqdm(patients, desc="Rebuilding remaining 3D Meshes"):
        if pid == target_pid or pid == "Lung_Dx-A0164":
            continue
            
        gltf_path = os.path.join(agent.processed_dir, pid, "tumor_mesh.gltf")
        # Just run them all to be safe and ensure dual-mesh is built
        try:
            agent.create_3d_snapshot(pid)
        except Exception as e:
            pass

if __name__ == "__main__":
    main()
