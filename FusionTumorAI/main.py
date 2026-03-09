
import argparse
import sys
import os
import logging
import json

# Add agents path
sys.path.append(os.path.join(os.path.dirname(__file__), 'agents'))

# Imports moved inside main() for robustness against missing dependencies

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/system.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    parser = argparse.ArgumentParser(description="FusionTumorAI - Master Orchestration")
    parser.add_argument("--step", type=str, default="all", 
                        choices=["all", "explore", "preprocess", "register", "mask", "roi", "patch", "train", "inference", "radiomics", "classify", "explain", "visualize", "report"],
                        help="Pipeline step to execute")
    parser.add_argument("--config", type=str, default="configs/config.json", help="Path to config file")
    parser.add_argument("--id", type=str, default=None, help="Run pipeline for a single patient ID")
    
    args = parser.parse_args()
    
    print("Initializing FusionTumorAI System...")
    logging.info(f"System started with step: {args.step}, patient: {args.id}")

    # Helper function to run batch or single
    def run_agent(agent):
        if args.id:
            # We assume agents have a method to process single patients if they handle batches
            # For data exploration/training, this might not apply cleanly, 
            # but for processing agents (inference, visualize, report), it's useful.
            try:
                if hasattr(agent, 'extract_features'): agent.extract_features(args.id)
                elif hasattr(agent, 'create_3d_snapshot'): agent.create_3d_snapshot(args.id)
                elif hasattr(agent, 'generate_report'): agent.generate_report(args.id)
                elif hasattr(agent, 'process_patient'): agent.process_patient(args.id)
                elif hasattr(agent, 'infer_patient'): agent.infer_patient(args.id)
                else: 
                     print(f"Agent {agent.__class__.__name__} doesn't support single patient execution directly via main. Running batch.")
                     agent.run_batch()
            except Exception as e:
                print(f"Failed single execution: {e}")
                agent.run_batch() # Fallback
        else:
            agent.run_batch()

    # 1. Exploration
    if args.step in ["all", "explore"]:
        from agents.dataset_explorer import DatasetExplorerAgent
        print("\n[1/12] Running Dataset Explorer...")
        agent = DatasetExplorerAgent(args.config)
        df = agent.scan_dataset()
        val_df = agent.validate_pairing(df)
        agent.generate_report(val_df, df)

    # 2. Preprocessing
    if args.step in ["all", "preprocess"]:
        from agents.preprocessing import DICOMPreprocessingAgent
        print("\n[2/12] Running DICOM Preprocessing...")
        agent = DICOMPreprocessingAgent(args.config)
        agent.run_batch("metadata.csv")

    # 3. Registration
    if args.step in ["all", "register"]:
        from agents.registration import RegistrationAgent
        print("\n[3/12] Running Registration...")
        agent = RegistrationAgent(args.config)
        run_agent(agent)

    # Mask Parsing (Extra)
    if args.step in ["all", "mask"]:
        from agents.mask_parser import MaskParser
        print("\n[Extra] Parsing XML Masks...")
        agent = MaskParser(args.config)
        run_agent(agent)

    # 4. ROI Extraction
    if args.step in ["all", "roi"]:
        from agents.roi import LungROIExtractionAgent
        print("\n[4/12] Running ROI Extraction...")
        agent = LungROIExtractionAgent(args.config)
        run_agent(agent)

    # 5. Patch Generation
    if args.step in ["all", "patch"]:
        from agents.patch_generator import PatchGeneratorAgent
        print("\n[5/12] Generating Patches...")
        agent = PatchGeneratorAgent(args.config)
        run_agent(agent)

    # 6. Training
    if args.step in ["all", "train"]:
        from agents.train_segmentation import SegmentationTrainingAgent
        print("\n[6/12] Training Segmentation Model...")
        agent = SegmentationTrainingAgent(args.config)
        if not args.id: # Don't train on a single patient run
            agent.train(epochs=2) 

    # 7. Inference
    if args.step in ["all", "inference"]:
        from agents.inference import InferenceAgent
        print("\n[7/12] Running Inference...")
        agent = InferenceAgent(args.config)
        if args.id: agent.infer_patient(args.id)
        else: agent.run_batch()

    # 8. Radiomics
    if args.step in ["all", "radiomics"]:
        from agents.radiomics import RadiomicsAgent
        print("\n[8/12] Extracting Radiomics...")
        agent = RadiomicsAgent(args.config)
        run_agent(agent)

    # 9. Classification
    if args.step in ["all", "classify"]:
        from agents.classifier import FusionClassifierAgent
        print("\n[9/12] Training Classifier...")
        agent = FusionClassifierAgent(args.config)
        if not args.id:
            agent.train()

    # 10. Explainability
    if args.step in ["all", "explain"]:
        from agents.explainability import ExplainabilityAgent
        print("\n[10/12] Generating Explainability Maps...")
        agent = ExplainabilityAgent(args.config)
        run_agent(agent)

    # 11. Visualization
    if args.step in ["all", "visualize"]:
        from agents.visualization import VisualizationAgent
        print("\n[11/12] Creating Visualizations...")
        agent = VisualizationAgent(args.config)
        run_agent(agent)

    # 12. Reporting
    if args.step in ["all", "report"]:
        from agents.report_generator import ReportGenerationAgent
        print("\n[12/12] Generating Reports...")
        agent = ReportGenerationAgent(args.config)
        run_agent(agent)

    print("\n✅ Execution Complete.")
    logging.info("Execution Complete.")

if __name__ == "__main__":
    main()
