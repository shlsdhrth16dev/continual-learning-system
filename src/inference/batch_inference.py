import logging
import json
from pathlib import Path
from datetime import datetime

import joblib
import pandas as pd

from src.registry.model_registry import load_model, get_latest_version

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = Path("data/predictions")
INFERENCE_DATA_DIR = Path("data/inference")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")

def load_inference_artifacts():
    """Loads the latest model and the preprocessor."""
    latest_v = get_latest_version()
    if not latest_v:
        raise RuntimeError("No model versions found in registry. Run training first.")
    
    logger.info(f"Loading model version {latest_v}...")
    model, metadata = load_model(latest_v)
    
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}")
    
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor, latest_v

def run_inference(batch_path: Path):
    """Runs inference on a single batch file."""
    model, preprocessor, version = load_inference_artifacts()
    
    logger.info(f"Loading batch data from {batch_path}...")
    df_raw = pd.read_csv(batch_path)
    
    # Drop target if present
    features_raw = df_raw.drop(columns=["default", "default.payment.next.month"], errors="ignore")
    
    logger.info("Preprocessing features...")
    features_processed = preprocessor.transform(features_raw)
    
    logger.info("Generating predictions...")
    preds = model.predict(features_processed)
    
    # Construct results
    result_df = features_raw.copy()
    result_df["prediction"] = preds
    result_df["model_version"] = version
    result_df["inference_time"] = datetime.utcnow().isoformat()
    
    return result_df

def main():
    INFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    
    batch_files = sorted(INFERENCE_DATA_DIR.glob("*.csv"))
    if not batch_files:
        logger.warning(f"No inference batches found in {INFERENCE_DATA_DIR}")
        return

    for batch_path in batch_files:
        try:
            logger.info(f"Processing batch: {batch_path.name}")
            result_df = run_inference(batch_path)
            
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = PREDICTIONS_DIR / f"preds_{batch_path.stem}_{timestamp}.csv"
            
            result_df.to_csv(output_path, index=False)
            logger.info(f"Predictions saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to process batch {batch_path.name}: {e}")

if __name__ == "__main__":
    main()
