import os
import json
from datetime import datetime

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.training.data_loader import load_data
from src.registry.model_registry import register_model
from src.drift.reference_stats import compute_reference_stats

MODEL_VERSION = "v1"



import os
import json
import logging
from datetime import datetime

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.training.data_loader import load_data
from src.registry.model_registry import register_model
from src.drift.reference_stats import compute_reference_stats

MODEL_VERSION = "v1"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Load data
    logger.info("Loading data...")
    X_raw, y = load_data()

    # 2. Load Preprocessor (Created in Phase 1)
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}. Run src/data/preprocessor.py first.")
    
    logger.info("Loading preprocessor...")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # 3. Transform Features
    logger.info("Transforming features...")
    # Note: X_raw might need column alignment if preprocessor expects specific columns
    # In Phase 1, we saved preprocessor fitted on this exact schema.
    X_processed = preprocessor.transform(X_raw)

    # 4. Train / Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_processed, y, test_size=0.2, random_state=42
    )

    # 5. Train baseline model
    logger.info("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluate
    preds = model.predict(X_val)
    accuracy = accuracy_score(y_val, preds)
    logger.info(f"Validation Accuracy: {accuracy:.4f}")

    # 7. Metadata
    metadata = {
        "model_version": MODEL_VERSION,
        "algorithm": "logistic_regression",
        "metrics": {
            "accuracy": accuracy
        },
        "features": list(X_raw.columns), # Store raw feature names for reference
        "trained_at": datetime.utcnow().isoformat()
    }

    # 8. Register model
    register_model(model, metadata, MODEL_VERSION)

    # 9. Compute reference feature stats (using Raw X_train or Processed?)
    # Usually drift detection is on RAW incoming features (Data Drift) 
    # OR on Model Outputs (Concept Drift). 
    # Let's save RAW stats for Data Drift.
    stats = compute_reference_stats(X_raw)

    os.makedirs("data/reference", exist_ok=True)
    with open(f"data/reference/feature_stats_{MODEL_VERSION}.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("✅ Phase 2.2 complete")
    
if __name__ == "__main__":
    main()

