# FusionTumorAI Kubeflow Deployment

This folder contains the files necessary to deploy your `FusionTumorAI` project into a Kubeflow cluster.

## 1. Build and Push the Docker Image
Kubeflow runs your code inside Docker containers. You must first build the image and push it to a container registry that your Kubeflow installation can access (e.g., Docker Hub, Google Container Registry).

Run these commands from the **root** of your project (one folder up):

```bash
cd ..
# Build the image
docker build -t duck8bit/fusiontumorai:latest -f kubeflow/Dockerfile .

# Push the image to the registry
docker push duck8bit/fusiontumorai:latest
```

## 2. Compile the Pipeline
The `pipeline.py` script defines the sequence of your agents (Preprocessing -> Segmentation -> Radiomics -> Classifier -> Visualization -> Report).

First, ensure you have the Kubeflow Pipelines SDK installed:
```bash
pip install kfp
```

Next, edit `kubeflow/pipeline.py` and change the `DOCKER_IMAGE` variable at the top of the file to match the image name you just built and pushed (`yourusername/fusiontumorai:latest`).

Then, compile the pipeline into a `.yaml` file:
```bash
cd kubeflow
python pipeline.py
```
This will generate `fusion_tumor_pipeline.yaml` in this directory.

## 3. Upload to Kubeflow Dashboard
1. Open your Kubeflow Pipelines UI (the dashboard you shared).
2. Click the **+ Upload pipeline** button in the top right.
3. Select the `fusion_tumor_pipeline.yaml` file you just generated.
4. Give it a name and click upload.
5. You can now click **Create Run** to execute the ML pipeline!
