"""
Automated monitoring pipeline that integrates drift detection with retraining.
Run this script on a schedule (e.g., daily via cron) to maintain model performance.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from src.monitoring.drift_detector import detect_drift, load_latest_reference_stats
from src.training.retrain import retrain_pipeline, should_trigger_retraining
from src.utils.paths import INFERENCE_DATA_DIR, DRIFT_REPORTS_DIR, ensure_directory
from src.utils.logging_config import get_logger
from src.config import get_config

logger = get_logger(__name__)


def run_drift_detection() -> dict:
    """
    Run drift detection on available inference data.
    
    Returns:
        Drift report dictionary
    """
    logger.info("Running drift detection...")
    
    # Find latest inference batch
    batch_files = sorted(INFERENCE_DATA_DIR.glob("*.csv"))
    if not batch_files:
        logger.warning(f"No inference data found in {INFERENCE_DATA_DIR}")
        return None
    
    latest_batch = batch_files[-1]
    logger.info(f"Analyzing drift for {latest_batch.name}")
    
    # Load reference stats
    try:
        import pandas as pd
        ref_stats, version = load_latest_reference_stats()
        batch_df = pd.read_csv(latest_batch)
        
        # Detect drift
        drifted_features = detect_drift(ref_stats, batch_df)
        
        # Determine severity
        severity = "none"
        if drifted_features:
            # Check for high PSI values
            max_psi = max(
                f.get("metrics", {}).get("psi", 0)
                for f in drifted_features.values()
            )
            max_z = max(
                f.get("metrics", {}).get("z_score", 0)
                for f in drifted_features.values()
            )
            
            if max_psi > 0.5 or max_z > 5:
                severity = "critical"
            elif max_psi > 0.25 or max_z > 4:
                severity = "high"
            else:
                severity = "moderate"
        
        drift_report = {
            "batch_name": latest_batch.stem,
            "model_version": version,
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "drifted_features": drifted_features,
            "n_drifted_features": len(drifted_features)
        }
        
        # Save report
        ensure_directory(DRIFT_REPORTS_DIR)
        report_path = DRIFT_REPORTS_DIR / f"drift_report_{latest_batch.stem}.json"
        with open(report_path, "w") as f:
            json.dump(drift_report, f, indent=2)
        
        logger.info(f"Drift detection complete. Severity: {severity}, Report: {report_path}")
        
        return drift_report
        
    except Exception as e:
        logger.error(f"Drift detection failed: {e}", exc_info=True)
        return None


def main():
    """Main monitoring pipeline."""
    parser = argparse.ArgumentParser(description="Automated monitoring and retraining pipeline")
    parser.add_argument("--dry-run", action="store_true", help="Run drift detection without triggering retraining")
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining regardless of drift")
    parser.add_argument("--data-path", type=str, help="Path to training data (overrides default)")
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("AUTOMATED MONITORING PIPELINE")
    logger.info(f"Started at: {datetime.now().isoformat()}")
    logger.info("="*80)
    
    config = get_config()
    
    # Step 1: Run drift detection
    drift_report = run_drift_detection()
    
    if drift_report:
        logger.info(f"Drift analysis: severity={drift_report['severity']}, "
                   f"n_drifted={drift_report['n_drifted_features']}")
    
    # Step 2: Decide on retraining
    should_retrain = False
    
    if args.force_retrain:
        logger.info("Forced retraining requested")
        should_retrain = True
    elif drift_report and should_trigger_retraining(drift_report):
        should_retrain = True
    
    # Step 3: Execute retraining if needed
    if should_retrain:
        if args.dry_run:
            logger.info("DRY RUN: Would trigger retraining but skipping due to --dry-run flag")
        else:
            logger.info("Triggering automated retraining...")
            
            data_path = Path(args.data_path) if args.data_path else None
            result = retrain_pipeline(data_path=data_path, drift_info=drift_report)
            
            if result['success']:
                logger.info(f"Retraining successful: {result['outcome']}")
                logger.info(f"New model: {result['new_version']}")
                logger.info(f"Metrics: {result['metrics']}")
            else:
                logger.error(f"Retraining failed: {result.get('error')}")
    else:
        logger.info("No retraining triggered - drift levels within acceptable range")
    
    logger.info("="*80)
    logger.info("Monitoring pipeline complete")
    logger.info("="*80)


if __name__ == "__main__":
    main()
