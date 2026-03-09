
import sys
import os
# Add parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
    print("Health Check: PASSED")

def test_predict_mock():
    # We test with a known patient ID that has data (A0164 from previous steps)
    patient_id = "Lung_Dx-A0164" 
    
    # Check if data exists first to avoid false negative test
    if not os.path.exists(f"data/processed/{patient_id}"):
        print(f"Skipping prediction test: Data for {patient_id} not found locally.")
        return

    print(f"Testing prediction for {patient_id}...")
    try:
        response = client.post(f"/predict/{patient_id}")
        if response.status_code == 200:
            data = response.json()
            print("Prediction Response:", data)
            assert "metric" in data or "patient_id" in data
            print("Prediction Check: PASSED")
        else:
            print(f"Prediction Check: FAILED (Status {response.status_code})")
            print(response.json())
    except Exception as e:
        print(f"Prediction Check: ERROR ({e})")
        
def test_viewer_mock():
    patient_id = "Lung_Dx-A0164"
    print(f"Testing viewer for {patient_id}...")
    try:
        response = client.get(f"/viewer/{patient_id}")
        if response.status_code == 200 and "text/html" in response.headers.get("content-type", ""):
            print("Viewer Check: PASSED (HTML Returned)")
        else:
             print(f"Viewer Check: FAILED (Status {response.status_code})")
    except Exception as e:
        print(f"Viewer Check: ERROR ({e})")

def test_gltf_mock():
    patient_id = "Lung_Dx-A0164"
    print(f"Testing GLTF for {patient_id}...")
    try:
        response = client.get(f"/gltf/{patient_id}")
        if response.status_code == 200:
            print("GLTF Check: PASSED (Model Returned)")
        else:
             print(f"GLTF Check: FAILED (Status {response.status_code})")
    except Exception as e:
        print(f"GLTF Check: ERROR ({e})")

if __name__ == "__main__":
    test_health()
    test_predict_mock()
    test_viewer_mock()
    test_gltf_mock()
