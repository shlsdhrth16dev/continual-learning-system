"""
Automated retraining orchestrator with sophisticated promotion logic.
Integrates drift detection with model retraining decisions.
"""
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.training.trainer import train_model
from src.training.evaluator import evaluate_model
from src.registry.model_registry import (
    register_model, promote_model, get_production_model,
    rollback_to_version, RegistryError
)
from src.config import get_config
from src.utils.paths import PROCESSED_DATA_DIR
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RetrainingDecision:
    """Encapsulates a retraining decision with reasoning."""
    
    def __init__(self, should_promote: bool, reason: str, details: Dict[str, Any]):
        self.should_promote = should_promote
        self.reason = reason
        self.details = details
    
    def __str__(self):
        status = "✅ PROMOTE" if self.should_promote else "❌ REJECT"
        return f"{status}: {self.reason}"


def make_promotion_decision(
    eval_results: Dict[str, Any],
    metadata: Dict[str, Any]
) -> RetrainingDecision:
    """
    Make intelligent promotion decision based on evaluation results.
    
    Args:
        eval_results: Evaluation results from evaluator
        metadata: Training metadata
    
    Returns:
        RetrainingDecision object
    """
    config = get_config()
    metrics = eval_results['metrics']
    comparison = eval_results.get('comparison_with_production')
    
    primary_metric = config.retraining.primary_metric
    
    # If no production model, promote if metrics are reasonable
    if comparison is None:
        if primary_metric in metrics and metrics[primary_metric] > 0.5:
            return RetrainingDecision(
                should_promote=True,
                reason="No existing production model. New model has acceptable performance.",
                details={"metrics": metrics}
            )
        else:
            return RetrainingDecision(
                should_promote=False,
                reason=f"Low {primary_metric}: {metrics.get(primary_metric, 0):.4f}",
                details={"metrics": metrics}
            )
    
    # Compare with production model
    improvement = comparison['improvement']['absolute']
    is_significant = comparison['statistical_test']['is_significant']
    p_value = comparison['statistical_test']['p_value']
    
    # Decision logic
    if improvement >= config.retraining.min_improvement_threshold:
        if config.retraining.require_significance:
            if is_significant:
                return RetrainingDecision(
                    should_promote=True,
                    reason=f"Significant improvement in {primary_metric}: +{improvement:.4f} (p={p_value:.4f})",
                    details=comparison
                )
            else:
                return RetrainingDecision(
                    should_promote=False,
                    reason=f"Improvement not statistically significant (p={p_value:.4f})",
                    details=comparison
                )
        else:
            return RetrainingDecision(
                should_promote=True,
                reason=f"Improvement in {primary_metric}: +{improvement:.4f}",
                details=comparison
            )
    
    elif improvement >= -config.retraining.max_degradation_threshold:
        # Small degradation but within acceptable range
        return RetrainingDecision(
            should_promote=False,
            reason=f"Marginal performance change: {improvement:+.4f} (within tolerance)",
            details=comparison
        )
    
    else:
        # Significant degradation
        return RetrainingDecision(
            should_promote=False,
            reason=f"Performance degradation: {improvement:.4f} in {primary_metric}",
            details=comparison
        )


def retrain_pipeline(
    data_path: Optional[Path] = None,
    target_col: str = "target",
    drift_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Execute the complete retraining pipeline with smart promotion logic.
    
    Args:
        data_path: Path to training data (defaults to latest processed batch)
        target_col: Target column name
        drift_info: Optional drift detection information
    
    Returns:
        Dictionary with retraining results
    """
    config = get_config()
    logger.info("="*80)
    logger.info("STARTING RETRAINING PIPELINE")
    logger.info("="*80)
    
    start_time = datetime.now()
    
    try:
        # Determine data path
        if data_path is None:
            data_path = PROCESSED_DATA_DIR / "latest_batch.csv"
            logger.info(f"Using default data path: {data_path}")
        
        if not data_path.exists():
            raise FileNotFoundError(f"Training data not found at {data_path}")
        
        # Log drift information if provided
        if drift_info:
            logger.info(f"Triggered by drift detection: severity={drift_info.get('severity')}")
            logger.info(f"Drifted features: {list(drift_info.get('drifted_features', {}).keys())}")
        
        # Get current production model info
        prod_model = get_production_model()
        if prod_model:
            logger.info(f"Current production model: {prod_model['version']} (metrics: {prod_model['metrics']})")
        else:
            logger.info("No current production model")
        
        # Register placeholder for new model (gets version number)
        new_version = register_model(
            metrics={},
            metadata={"triggered_by": "drift" if drift_info else "manual"},
            config=config.model.to_dict()
        )
        logger.info(f"Registered new candidate: {new_version}")
        
        # Train model
        logger.info(f"Training {new_version}...")
        model, train_metadata = train_model(data_path, new_version, target_col)
        
        # Evaluate model
        logger.info(f"Evaluating {new_version}...")
        eval_results = evaluate_model(model, data_path, new_version, use_test_set=True)
        
        # Make promotion decision
        decision = make_promotion_decision(eval_results, train_metadata)
        logger.info(f"Promotion decision: {decision}")
        
        # Update registry with actual metrics
        from src.registry.model_registry import load_registry, save_registry
        registry = load_registry()
        for model_entry in registry['history']:
            if model_entry['version'] == new_version:
                model_entry['metrics'] = eval_results['metrics']
                model_entry['metadata'].update(train_metadata)
                if drift_info:
                    model_entry['metadata']['drift_info'] = drift_info
                break
        save_registry(registry)
        
        # Execute decision
        if decision.should_promote:
            promote_model(new_version, reason=decision.reason)
            logger.info(f"✅ PROMOTED {new_version} to production")
            outcome = "promoted"
        else:
            logger.info(f"❌ NOT PROMOTED: {decision.reason}")
            outcome = "rejected"
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        result = {
            "success": True,
            "outcome": outcome,
            "new_version": new_version,
            "metrics": eval_results['metrics'],
            "decision": {
                "should_promote": decision.should_promote,
                "reason": decision.reason,
                "details": decision.details
            },
            "duration_seconds": duration,
            "timestamp": end_time.isoformat()
        }
        
        logger.info(f"Retraining pipeline completed in {duration:.1f}s")
        logger.info("="*80)
        
        return result
        
    except Exception as e:
        logger.error(f"Retraining pipeline failed: {e}", exc_info=True)
        
        # Try to rollback if we promoted a bad model
        if prod_model and 'new_version' in locals():
            try:
                logger.warning("Attempting rollback to previous production model")
                rollback_to_version(prod_model['version'])
            except Exception as rollback_error:
                logger.error(f"Rollback failed: {rollback_error}")
        
        return {
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def should_trigger_retraining(drift_report: Dict[str, Any]) -> bool:
    """
    Determine if retraining should be triggered based on drift report.
    
    Args:
        drift_report: Drift detection report
    
    Returns:
        True if retraining should be triggered
    """
    config = get_config()
    
    if not config.drift.enable_auto_retraining:
        logger.info("Auto-retraining is disabled")
        return False
    
    severity = drift_report.get('severity', 'none')
    drifted_features = drift_report.get('drifted_features', {})
    n_drifted = len(drifted_features)
    
    # Trigger conditions
    if severity == 'critical':
        logger.info(f"Critical drift detected -> triggering retraining")
        return True
    
    if severity == 'high' and n_drifted >= config.drift.min_drifted_features_high:
        logger.info(f"High severity drift with {n_drifted} features -> triggering retraining")
        return True
    
    if severity == 'moderate' and n_drifted >= config.drift.min_drifted_features_moderate:
        logger.info(f"Moderate drift with {n_drifted} features -> triggering retraining")
        return True
    
    logger.info(f"Drift severity '{severity}' with {n_drifted} features -> no retraining needed")
    return False
