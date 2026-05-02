import kfp
from kfp import dsl
from kfp import compiler

# Define the Docker image that contains your codebase
# NOTE: Once you build the Dockerfile, push it to a registry and replace this URL
DOCKER_IMAGE = "docker.io/duck8bit/fusiontumorai:latest"

# Define the Pipeline architecture (DAG) using V1 syntax (ContainerOp)
@dsl.pipeline(
    name="fusion-tumor-ai",
    description="End-to-End Lung Cancer Detection and Analysis Pipeline"
)
def fusion_tumor_pipeline():
    # Mount a Persistent Volume Claim (PVC) to share the 'data/' folder across steps
    # Kubeflow runs each step in an isolated container. They must share a disk.
    pvc = dsl.VolumeOp(
        name="fusion-data-volume",
        resource_name="fusion-pvc",
        size="20Gi",
        modes=dsl.VOLUME_MODE_RWM
    )
    
    # Define execution order using ContainerOps (creates proper v1alpha1 Argo YAML)
    preprocess = dsl.ContainerOp(
        name="Preprocessing",
        image=DOCKER_IMAGE,
        command=["python", "agents/preprocessing.py"]
    ).add_pvolumes({"/app/data": pvc.volume})
    
    segmentation = dsl.ContainerOp(
        name="Segmentation",
        image=DOCKER_IMAGE,
        command=["python", "agents/inference.py"]
    ).add_pvolumes({"/app/data": pvc.volume}).after(preprocess)
    
    radiomics = dsl.ContainerOp(
        name="Radiomics Extraction",
        image=DOCKER_IMAGE,
        command=["python", "agents/radiomics.py"]
    ).add_pvolumes({"/app/data": pvc.volume}).after(segmentation)
    
    classifier = dsl.ContainerOp(
        name="Classifier",
        image=DOCKER_IMAGE,
        command=["python", "agents/classifier.py"]
    ).add_pvolumes({"/app/data": pvc.volume}).after(radiomics)
    
    visualization = dsl.ContainerOp(
        name="Visualization",
        image=DOCKER_IMAGE,
        command=["python", "agents/visualization.py"]
    ).add_pvolumes({"/app/data": pvc.volume}).after(classifier)

    reporting = dsl.ContainerOp(
        name="Report Generation",
        image=DOCKER_IMAGE,
        command=["python", "agents/report_generator.py"]
    ).add_pvolumes({"/app/data": pvc.volume}).after(visualization)

    # Satisfy kf-resource-quota by forcefully applying resource limits to all execution steps
    for step in [preprocess, segmentation, radiomics, classifier, visualization, reporting]:
        step.set_cpu_request('100m')
        step.set_cpu_limit('2')
        step.set_memory_request('500Mi')
        step.set_memory_limit('4Gi')
        step.add_pod_annotation('sidecar.istio.io/inject', 'false')


if __name__ == "__main__":
    # Compile the pipeline loop into an uploadable YAML file
    pipeline_filename = "fusion_tumor_pipeline_v1.yaml"
    compiler.Compiler().compile(
        pipeline_func=fusion_tumor_pipeline,
        package_path=pipeline_filename
    )
    print(f"✅ Pipeline compiled successfully into {pipeline_filename}")
    print(f"You can now upload '{pipeline_filename}' to the Kubeflow dashboard.")
