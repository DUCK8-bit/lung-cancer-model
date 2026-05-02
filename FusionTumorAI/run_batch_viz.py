"""
Subprocess-isolated batch runner for visualization.
Each patient is processed in a separate subprocess so a VTK/PyVista
segfault on one patient doesn't kill the entire batch.
"""
import os
import sys
import subprocess
import time

PROCESSED_DIR = "data/processed"

def get_patients():
    return sorted([
        d for d in os.listdir(PROCESSED_DIR)
        if os.path.isdir(os.path.join(PROCESSED_DIR, d))
    ])

def needs_update(patient_id):
    """Check if a patient's 3d_viewer.html was updated in the last 2 hours."""
    html = os.path.join(PROCESSED_DIR, patient_id, "3d_viewer.html")
    if not os.path.exists(html):
        return True
    return (time.time() - os.path.getmtime(html)) > 7200

def process_patient(patient_id):
    """Run visualization for a single patient in a subprocess."""
    code = f"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
sys.path.insert(0, ".")
from agents.visualization import VisualizationAgent
ag = VisualizationAgent()
ag.create_3d_snapshot("{patient_id}")
print("DONE:{patient_id}")
"""
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if "DONE:" in result.stdout:
            return True, None
        else:
            err = result.stderr[-500:] if result.stderr else "Unknown error"
            return False, err
    except subprocess.TimeoutExpired:
        return False, "Timeout (>5min)"
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    patients = get_patients()
    # Filter to only patients needing update (pass --all to force all)
    if "--all" not in sys.argv:
        patients = [p for p in patients if needs_update(p)]
    
    total = len(patients)
    print(f"Processing {total} patients...")
    
    success = 0
    failed = []
    for i, pid in enumerate(patients):
        print(f"[{i+1}/{total}] {pid}...", end=" ", flush=True)
        ok, err = process_patient(pid)
        if ok:
            print("OK")
            success += 1
        else:
            print(f"FAILED: {err[:100]}")
            failed.append((pid, err))
    
    print(f"\n{'='*50}")
    print(f"Results: {success}/{total} succeeded")
    if failed:
        print(f"Failed patients ({len(failed)}):")
        for pid, err in failed:
            print(f"  {pid}: {err[:150]}")
    print("Batch visualization complete.")
