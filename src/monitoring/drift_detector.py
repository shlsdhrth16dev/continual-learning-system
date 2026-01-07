import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd

from src.registry.model_registry import get_latest_version

# Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

REFERENCE_STATS_DIR = Path("data/reference")
DRIFT_REPORTS_DIR = Path("data/drift_reports")
INFERENCE_DATA_DIR = Path("data/inference")

PSI_THRESHOLD = 0.2
Z_SCORE_THRESHOLD = 3.0

def load_latest_reference_stats() -> Tuple[Dict[str, Any], str]:
    """Loads reference stats matching the latest model version."""
    latest_v = get_latest_version()
    if not latest_v:
        raise RuntimeError("No model versions found in registry.")
    
    stats_path = REFERENCE_STATS_DIR / f"feature_stats_{latest_v}.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Reference stats not found at {stats_path}")
    
    with open(stats_path, "r") as f:
        return json.load(f), latest_v

def calculate_psi(expected_dist: Dict[str, float], actual_values: pd.Series) -> float:
    """
    Calculates Population Stability Index (PSI).
    For numeric features, this implementation uses a simplified approach if full buckets aren't available.
    For categoricals, it compares proportions.
    """
    actual_dist = actual_values.value_counts(normalize=True).to_dict()
    
    # Ensure both distributions have all keys
    all_keys = set(expected_dist.keys()).union(set(actual_dist.keys()))
    
    psi = 0.0
    for key in all_keys:
        # Avoid zero-division or log(0)
        e = expected_dist.get(key, 0.0001)
        a = actual_dist.get(key, 0.0001)
        
        psi += (a - e) * np.log(a / e)
    
    return psi

def detect_drift(reference_stats: Dict[str, Any], batch_df: pd.DataFrame) -> Dict[str, Any]:
    """Detects drift across all features in the batch."""
    drift_report = {}
    
    for feature, ref in reference_stats.items():
        if feature not in batch_df.columns:
            continue
            
        batch_values = batch_df[feature].dropna()
        if batch_values.empty:
            continue
            
        feature_drift = {"drift_detected": False, "metrics": {}}
        
        if ref["type"] == "numeric":
            ref_stats = ref["stats"]
            batch_mean = batch_values.mean()
            z_score = abs(batch_mean - ref_stats["mean"]) / (ref_stats["std"] + 1e-8)
            
            feature_drift["metrics"]["z_score"] = float(z_score)
            if z_score > Z_SCORE_THRESHOLD:
                feature_drift["drift_detected"] = True
                
        elif ref["type"] == "categorical":
            ref_counts = ref["stats"]["value_counts"]
            # Convert counts to proportions for PSI
            total_ref = sum(ref_counts.values())
            ref_dist = {k: v/total_ref for k, v in ref_counts.items()}
            
            psi = calculate_psi(ref_dist, batch_values)
            feature_drift["metrics"]["psi"] = float(psi)
            if psi > PSI_THRESHOLD:
                feature_drift["drift_detected"] = True
                
        if feature_drift["drift_detected"]:
            drift_report[feature] = feature_drift

    return drift_report

def main():
    INFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    
    batch_files = sorted(INFERENCE_DATA_DIR.glob("*.csv"))
    if not batch_files:
        logger.warning(f"No batches found in {INFERENCE_DATA_DIR}")
        return

    try:
        ref_stats, version = load_latest_reference_stats()
        
        for batch_path in batch_files:
            logger.info(f"Analyzing drift for {batch_path.name}...")
            batch_df = pd.read_csv(batch_path)
            
            drifted_features = detect_drift(ref_stats, batch_df)
            
            severity = "none"
            if drifted_features:
                severity = "moderate"
                if any(f.get("metrics", {}).get("psi", 0) > 0.5 for f in drifted_features.values()):
                    severity = "high"
            
            report = {
                "batch_name": batch_path.stem,
                "model_version": version,
                "timestamp": datetime.utcnow().isoformat(),
                "severity": severity,
                "drifted_features": drifted_features
            }
            
            report_path = DRIFT_REPORTS_DIR / f"drift_report_{batch_path.stem}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Drift analysis complete. Severity: {severity}. Report: {report_path}")
            
    except Exception as e:
        logger.error(f"Drift detection failed: {e}")

if __name__ == "__main__":
    main()
