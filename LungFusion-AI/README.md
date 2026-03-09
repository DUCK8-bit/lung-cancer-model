# LungFusion-AI 🫁

**Multi-Modal Lung Cancer Diagnosis & Visualization System**

LungFusion-AI is a deep learning system designed to analyze Lung PET-CT scans for nodule detection and malignancy classification. It features a Dual-Stream 3D U-Net architecture and an interactive web dashboard for visualizing results.

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have Python 3.8+ installed.

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup
This project supports the **Lung-PET-CT-Dx** dataset (DICOM format).

1.  Download a subset of the dataset (e.g., "Subject A") from [TCIA](https://wiki.cancerimagingarchive.net/display/Public/Lung-PET-CT-Dx) or Kaggle.
2.  Place the downloaded `Lung-PET-CT-Dx` folder structure inside a `data/` directory in the project root:
    ```
    LungFusion-AI/
    ├── data/
    │   └── manifest-xxx/
    │       └── Lung-PET-CT-Dx/
    │           ├── Lung_Dx-A0001/
    │           ├── Lung_Dx-A0002/
    │           └── ...
    ```

### 3. Training the Model (Local)
To train the AI model on your local machine using the data in `data/`:

```bash
python src/train.py
```
*   This will scan your `data/` folder for DICOM series.
*   It trains a **FusionUNet** model for 2 epochs (configurable in `train.py`).
*   **Output**: A saved model file named `fusion_unet.pth` will be created in the project root.

### 4. Running the Dashboard
Launch the interactive visualization tool:

```bash
streamlit run app.py
```
*   Open your browser to `http://localhost:8501`.

## 🖥️ Using the Dashboard

### **Status Indicators**
*   **✅ SYSTEM READY**: The `fusion_unet.pth` model was found and loaded. AI predictions will be real.
*   **⚠️ DEMO MODE**: No model file found. The system is running with synthetic data and a "Demo" red sphere for visualization.

### **How to Analyze a Scan**
1.  **Sidebar**: locate the "Upload CT Scan" widget.
2.  **Upload**: Drag and drop a `.nii.gz` (NIfTI) file or a `.zip` containing DICOM files.
    *   *Tip: You can use `generate_demo_data.py` to create a `demo_lung.nii.gz` if you don't have one handy.*
3.  **Visualization**:
    *   **2D Viewer**: Use the slider to scroll through axial slices. A **Red Overlay** indicates the AI's predicted tumor region.
    *   **3D Viewer**: A 3D interactive rendering of the lung volume. The predicted tumor is extracted as a 3D mesh and shown in red.

## 📂 Project Structure
*   `app.py`: Main Streamlit dashboard application.
*   `src/model.py`: PyTorch definition of the FusionUNet (Dual-Stream 3D U-Net).
*   `src/train.py`: Training script with real DICOM loading logic.
*   `src/data_processing.py`: Helper functions for DICOM loading and preprocessing.
*   `src/analysis.py`: Radiomics feature extraction engine.
