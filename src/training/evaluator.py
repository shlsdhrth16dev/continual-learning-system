"""
Model evaluation module with comprehensive metrics and statistical testing.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from scipy import stats

from src.config import get_config
from src.utils.paths import get_model_version_path
from src.utils.logging_config import get_logger
from src.registry.model_registry import get_production_model

logger = get_logger(__name__)


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (for ROC-AUC)
    
    Returns:
        Dictionary of metrics
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # Add ROC-AUC if probabilities provided
    if y_proba is not None:
        try:
            # For binary classification
            if len(np.unique(y_true)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # For multiclass
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
        except Exception as e:
            logger.warning(f"Could not calculate ROC-AUC: {e}")
    
    # Round all metrics
    return {k: round(v, 4) for k, v in metrics.items()}


def mcnemar_test(y_true: pd.Series, y_pred1: np.ndarray, y_pred2: np.ndarray) -> Tuple[float, float]:
    """
    McNemar's test for comparing two models.
    Tests if there's a significant difference in error rates.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
    
    Returns:
        (statistic, p-value)
    """
    # Create contingency table
    # n00: both wrong, n01: model1 right model2 wrong, etc.
    n01 = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    n10 = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    
    # McNemar's test statistic
    if n01 + n10 == 0:
        return 0.0, 1.0
    
    statistic = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)
    
    return statistic, p_value


def compare_with_production(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_version: str
) -> Optional[Dict[str, Any]]:
    """
    Compare new model with production model.
    
    Args:
        model: New model to evaluate
        X_test: Test features
        y_test: Test labels
        model_version: Version of new model
    
    Returns:
        Comparison results or None if no production model
    """
    config = get_config()
    prod_model_info = get_production_model()
    
    if not prod_model_info:
        logger.info("No production model to compare against")
        return None
    
    try:
        # Load production model
        import joblib
        prod_version = prod_model_info['version']
        prod_model_path = get_model_version_path(prod_version) / "model.pkl"
        prod_model = joblib.load(prod_model_path)
        
        # Get predictions from both models
        new_pred = model.predict(X_test)
        prod_pred = prod_model.predict(X_test)
        
        # Calculate metrics for both
        new_proba = model.predict_proba(X_test)
        prod_proba = prod_model.predict_proba(X_test)
        
        new_metrics = calculate_metrics(y_test, new_pred, new_proba)
        prod_metrics = calculate_metrics(y_test, prod_pred, prod_proba)
        
        # Statistical significance test
        statistic, p_value = mcnemar_test(y_test, prod_pred, new_pred)
        
        primary_metric = config.retraining.primary_metric
        improvement = new_metrics[primary_metric] - prod_metrics[primary_metric]
        
        comparison = {
            "production_version": prod_version,
            "production_metrics": prod_metrics,
            "new_metrics": new_metrics,
            "improvement": {
                "absolute": round(improvement, 4),
                "relative_pct": round((improvement / prod_metrics[primary_metric] * 100), 2) if prod_metrics[primary_metric] > 0 else None
            },
            "statistical_test": {
                "test": "mcnemar",
                "statistic": round(statistic, 4),
                "p_value": round(p_value, 4),
                "is_significant": p_value < config.retraining.significance_level
            }
        }
        
        logger.info(f"Comparison: {primary_metric} improvement = {improvement:.4f}, p-value = {p_value:.4f}")
        
        return comparison
        
    except Exception as e:
        logger.error(f"Failed to compare with production model: {e}")
        return None


def evaluate_model(
    model: Any,
    data_path: Path,
    model_version: str,
    use_test_set: bool = True
) -> Dict[str, Any]:
    """
    Evaluate model with comprehensive metrics and comparison.
    
    Args:
        model: Trained model
        data_path: Path to evaluation data (or will use saved test set)
        model_version: Model version being evaluated
        use_test_set: If True, use saved test set from training
    
    Returns:
        Evaluation results dictionary
    """
    config = get_config()
    logger.info(f"Evaluating model {model_version}")
    
    model_dir = get_model_version_path(model_version)
    
    # Load test data
    if use_test_set and (model_dir / "X_test.csv").exists():
        logger.info("Using saved test set for evaluation")
        X_test = pd.read_csv(model_dir / "X_test.csv")
        y_test = pd.read_csv(model_dir / "y_test.csv").squeeze()
    else:
        logger.info(f"Loading evaluation data from {data_path}")
        df = pd.read_csv(data_path)
        target_col = config.data.target_column
        X_test = df.drop(columns=[target_col])
        y_test = df[target_col]
    
    logger.info(f"Evaluating on {len(X_test)} samples")
    
    # Get predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Per-class report
    class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Compare with production
    comparison = compare_with_production(model, X_test, y_test, model_version)
    
    # Build results
    results = {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": class_report,
        "n_test_samples": len(X_test),
        "comparison_with_production": comparison
    }
    
    # Save results
    metrics_path = model_dir / "evaluation.json"
    with open(metrics_path, "w") as f:
        # Convert numpy types for JSON serialization
        json_results = json.loads(json.dumps(results, default=int))
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Evaluation complete. Metrics: {metrics}")
    logger.info(f"Results saved to {metrics_path}")
    
    return results

