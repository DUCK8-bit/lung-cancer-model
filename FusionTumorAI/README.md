# FusionTumorAI: Autonomous Multimodal Lung Analysis Agent

## Overview
FusionTumorAI is a modular, 12-agent system designed for end-to-end 3D PET-CT tumor analysis. It handles data exploration, preprocessing, segmentation (3D U-Net), radiomics extraction, malignancy classification, and automated reporting.

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.8+
- NVIDIA GPU with CUDA (Recommended for fast training)

### 2. Installation
Navgiate to the project folder and install dependencies:
```bash
cd FusionTumorAI
pip install -r requirements.txt
```

### 3. Usage
Run the master orchestration script `main.py` with specific steps:

**Full Pipeline:**
```bash
python main.py --step all
```

**Individual Steps:**
- Explore Data: `python main.py --step explore`
- Preprocessing: `python main.py --step preprocess`
- Registration: `python main.py --step register`
- ROI Extraction: `python main.py --step roi`
- Patch Generation: `python main.py --step patch`
- Training: `python main.py --step train`
- Inference: `python main.py --step inference`
- Radiomics: `python main.py --step radiomics`
- Classification: `python main.py --step classify`
- Explainability: `python main.py --step explain`
- Visualization: `python main.py --step visualize`
- Reporting: `python main.py --step report`

## 📂 Project Structure
- `agents/`: Source code for all 12 autonomous agents.
- `configs/`: Configuration files (paths, hyperparameters).
- `data/`: Dataset directory (raw, processed, patches).
- `models/`: Trained models (`unet_best.pth`, `classifier.pkl`).
- `reports/`: Generated PDF reports.
- `logs/`: Execution logs.

## ⚙️ Configuration
Modify `configs/config.json` to change dataset paths or hyperparameters.

## 🔧 Troubleshooting
- **Missing Dependencies**: Ensure you install from `requirements.txt`.
- **Memory Errors**: Reduce patch size in `config.json` or batch size in agents.
