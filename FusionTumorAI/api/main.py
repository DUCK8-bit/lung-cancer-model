from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uvicorn
from api.services import DiagnosticService

# Initialize App
app = FastAPI(
    title="FusionTumorAI Diagnostic API",
    description="Real-time Multi-modal Lung Nodule Classification & 3D Visualization API",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Business Logic
service = DiagnosticService()

@app.get("/")
def read_root():
    return {"status": "online", "system": "FusionTumorAI Diagnostic Server"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict/{patient_id}")
async def predict_patient(patient_id: str):
    """
    Runs the full inference pipeline for a patient.
    Returns JSON with tumor metrics and Infection Guard risk assessment.
    """
    try:
        results = service.run_prediction(patient_id)
        if not results:
             raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found or data missing.")
        return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/visualize/{patient_id}")
async def get_visualization(patient_id: str):
    """
    Generates and returns the 3D render of the tumor.
    """
    try:
        img_path = service.generate_visualization(patient_id)
        if not img_path:
             raise HTTPException(status_code=404, detail="Visualization could not be generated.")
        return FileResponse(img_path, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/viewer/{patient_id}")
async def get_interactive_viewer(patient_id: str):
    """
    Returns the standalone interactive 3D HTML viewer.
    """
    try:
        html_path = service.get_viewer(patient_id)
        if not html_path:
             raise HTTPException(status_code=404, detail="Interactive viewer not found. Run visualization first.")
        return FileResponse(html_path, media_type="text/html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/gltf/{patient_id}")
async def get_gltf_model(patient_id: str):
    """
    Returns the raw GLTF 3D model.
    """
    try:
        gltf_path = service.get_gltf(patient_id)
        if not gltf_path:
             raise HTTPException(status_code=404, detail="GLTF model not found.")
        return FileResponse(gltf_path, media_type="model/gltf+json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{patient_id}")
async def get_report(patient_id: str):
    """
    Generates and returns the diagnostic PDF report.
    """
    try:
        pdf_path = service.generate_report(patient_id)
        if not pdf_path:
             raise HTTPException(status_code=404, detail="Report could not be generated.")
        return FileResponse(pdf_path, media_type="application/pdf", filename=os.path.basename(pdf_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
