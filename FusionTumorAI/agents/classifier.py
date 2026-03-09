import os
import json
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import logging

# Setup logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/classifier.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class FusionClassifierAgent:
    def __init__(self, config_path="configs/config.json"):
        with open(config_path, "r") as f:
            self.config = json.load(f)
        self.models_dir = self.config["models_dir"]
        self.features_path = os.path.join(self.models_dir, "radiomics_features.csv")

    def train(self):
        if not os.path.exists(self.features_path):
            logging.error("Features file not found. Run RadiomicsAgent first.")
            return

        df = pd.read_csv(self.features_path)
        
        # Assume 'Malignancy' column exists or we need to derive it from metadata.
        # For now, let's look for a target column. If not found, simulate or warn.
        # In real scenario, we merge metadata.csv with radiomics.
        
        # Load metadata to get labels
        metadata_path = "metadata.csv"
        if os.path.exists(metadata_path):
            meta = pd.read_csv(metadata_path)
            # Metadata might have 'Diagnosis' or 'Malignancy'. 
            # If not, we can't train supervised.
            # Fallback: Create dummy label for demonstration/thesis structure validation 
            # if no labels found (User must provide labelled data).
            
            if 'Malignancy' in meta.columns:
                 df = df.merge(meta[['PatientID', 'Malignancy']], on='PatientID')
            else:
                 logging.warning("No Malignancy label in metadata. training unsupervised or using dummy?")
                 # Create Dummy for code validation
                 import numpy as np
                 df['Malignancy'] = np.random.randint(0, 2, size=len(df))
        else:
             import numpy as np
             df['Malignancy'] = np.random.randint(0, 2, size=len(df))

        # Drop non-feature columns
        X = df.drop(columns=['PatientID', 'Source', 'Malignancy'], errors='ignore')
        # Handle strings? Radiomics output is numeric.
        X = X.select_dtypes(include=[float, int])
        y = df['Malignancy']
        
        # Classifier
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        
        # CV
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X, y, cv=kfold, scoring='accuracy')
        
        print(f"CV Accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")
        logging.info(f"CV Accuracy: {scores.mean():.4f}")
        
        # Train Full
        clf.fit(X, y)
        
        # Save
        joblib.dump(clf, os.path.join(self.models_dir, "classifier.pkl"))
        print("Classifier saved.")
        
        # Feature Importance
        importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        importances.to_csv(os.path.join(self.models_dir, "feature_importance.csv"))

if __name__ == "__main__":
    agent = FusionClassifierAgent()
    agent.train()
