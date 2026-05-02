"""Process the 6 previously timed-out patients with skip_gif=True."""
import os, sys, subprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
failed = ["Lung_Dx-A0177","Lung_Dx-A0178","Lung_Dx-A0179","Lung_Dx-A0180","Lung_Dx-A0181","Lung_Dx-A0194"]

for i, pid in enumerate(failed):
    print(f"[{i+1}/{len(failed)}] Processing {pid} (skip_gif)...", end=" ", flush=True)
    code = (
        'import os; os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"; '
        'import sys; sys.path.insert(0,"."); '
        'from agents.visualization import VisualizationAgent; '
        'ag = VisualizationAgent(); '
        f'ag.create_3d_snapshot("{pid}", skip_gif=True); '
        f'print("DONE:{pid}")'
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True, text=True, timeout=600, cwd="."
        )
        if "DONE:" in result.stdout:
            print("OK")
        else:
            err_msg = result.stderr[-300:] if result.stderr else "Unknown"
            print(f"FAILED: {err_msg}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
    except Exception as e:
        print(f"ERROR: {e}")

print("All 6 patients processed.")
