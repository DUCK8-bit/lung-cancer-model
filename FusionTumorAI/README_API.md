# FusionTumorAI Diagnostic API Guide

## Overview
This API provides real-time access to the FusionTumorAI model, allowing external systems to request predictions, visualizations, and reports for lung cancer patients.

## Quick Start

### 1. Activate Environment
```bash
conda activate FusionTumorAI
```

### 2. Run the Server
 Navigate to the project root:
```bash
cd "c:\Users\ks181\Music\lung cancer model\FusionTumorAI"
```
Launch the API:
```bash
python api/main.py
```
*The server will start at `http://0.0.0.0:8000`*

## Endpoints

### 🩺 Health Check
- **URL:** `GET /health`
- **Response:** `{"status": "healthy"}`

### 🔮 Prediction & Infection Guard
- **URL:** `POST /predict/{patient_id}`
- **Example:** `http://localhost:8000/predict/Lung_Dx-A0164`
- **Response (JSON):**
    ```json
    {
        "patient_id": "Lung_Dx-A0164",
        "tumor_volume_cm3": 12.5,
        "max_suv": 8.4,
        "mean_hu": 45.0,
        "risk_assessment": {
            "status": "Risk: Malignancy",
            "description": "High Metabolic Activity (SUV 8.40) with Solid Tissue Density..."
        }
    }
    ```

### 🖼️ 3D Visualization
- **URL:** `GET /visualize/{patient_id}`
- **Example:** `http://localhost:8000/visualize/Lung_Dx-A0164`
- **Response:** Returns a PNG image of the 3D tumor render.

### 🌐 Interactive 3D Web Viewer (NASA-Style)
- **URL:** `GET /viewer/{patient_id}`
- **Example:** `http://localhost:8000/viewer/Lung_Dx-A0164`
- **Response:** Returns a standalone `3d_viewer.html` file. Can be opened in any browser for click-and-drag rotation, zooming, and map-like (X/Y/Z mm) measurement grids.

### 🧊 Raw 3D GLTF Model
- **URL:** `GET /gltf/{patient_id}`
- **Example:** `http://localhost:8000/gltf/Lung_Dx-A0164`
- **Response:** Returns `tumor_mesh.gltf` (the raw 3D mesh with embedded PET SUV intensities) for integration into custom frontend portals.

### 📄 PDF Report
- **URL:** `GET /report/{patient_id}`
- **Example:** `http://localhost:8000/report/Lung_Dx-A0164`
- **Response:** Downloads the full diagnostic PDF report.

## Interactive Documentation
Once the server is running, visit **[http://localhost:8000/docs](http://localhost:8000/docs)** for the auto-generated Swagger UI to test endpoints directly in your browser.
