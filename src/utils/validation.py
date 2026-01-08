"""
Validation utilities for data quality checks and model validation.
"""
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
import joblib

from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class ValidationError(Exception):
    """Raised when validation fails."""
    pass


def validate_dataframe_schema(
    df: pd.DataFrame,
    expected_columns: Optional[List[str]] = None,
    min_rows: int = 10
) -> Tuple[bool, str]:
    """
    Validate DataFrame schema and basic quality.
    
    Args:
        df: DataFrame to validate
        expected_columns: Expected column names (if None, just check basic quality)
        min_rows: Minimum number of rows required
    
    Returns:
        (is_valid, message) tuple
    """
    # Check minimum rows
    if len(df) < min_rows:
        return False, f"Insufficient data: {len(df)} rows (minimum: {min_rows})"
    
    # Check for expected columns
    if expected_columns is not None:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            return False, f"Missing columns: {missing_cols}"
        
        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            logger.warning(f"Extra columns found: {extra_cols}")
    
    # Check for all-null columns
    null_cols = df.columns[df.isnull().all()].tolist()
    if null_cols:
        return False, f"All-null columns found: {null_cols}"
    
    # Check for high null percentage
    high_null_cols = []
    for col in df.columns:
        null_pct = df[col].isnull().sum() / len(df)
        if null_pct > 0.9:
            high_null_cols.append(f"{col} ({null_pct:.1%})")
    
    if high_null_cols:
        logger.warning(f"Columns with >90% nulls: {high_null_cols}")
    
    # Check for duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        logger.warning(f"Found {dup_count} duplicate rows")
    
    return True, "Validation passed"


def validate_target_distribution(
    y: pd.Series,
    min_class_samples: int = 5
) -> Tuple[bool, str]:
    """
    Validate target variable distribution.
    
    Args:
        y: Target series
        min_class_samples: Minimum samples per class
    
    Returns:
        (is_valid, message) tuple
    """
    value_counts = y.value_counts()
    
    # Check minimum samples per class
    low_count_classes = value_counts[value_counts < min_class_samples]
    if len(low_count_classes) > 0:
        return False, f"Classes with <{min_class_samples} samples: {low_count_classes.to_dict()}"
    
    # Check for extreme imbalance
    max_count = value_counts.max()
    min_count = value_counts.min()
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 100:
        logger.warning(f"Extreme class imbalance: {imbalance_ratio:.1f}:1")
    
    logger.info(f"Target distribution: {value_counts.to_dict()}")
    
    return True, "Target validation passed"


def compute_data_hash(df: pd.DataFrame) -> str:
    """
    Compute a hash of the dataframe for tracking data versions.
    
    Args:
        df: DataFrame to hash
    
    Returns:
        MD5 hash string
    """
    # Create a hash from the dataframe content
    df_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.md5(df_bytes).hexdigest()


def validate_model_artifact(model_path: Path) -> Tuple[bool, str]:
    """
    Validate that a model artifact can be loaded.
    
    Args:
        model_path: Path to model .pkl file
    
    Returns:
        (is_valid, message) tuple
    """
    if not model_path.exists():
        return False, f"Model file not found: {model_path}"
    
    try:
        model = joblib.load(model_path)
        
        # Check if model has required methods
        required_methods = ['predict', 'predict_proba']
        for method in required_methods:
            if not hasattr(model, method):
                return False, f"Model missing method: {method}"
        
        return True, "Model artifact valid"
        
    except Exception as e:
        return False, f"Failed to load model: {str(e)}"


def validate_feature_consistency(
    current_features: List[str],
    expected_features: List[str]
) -> Tuple[bool, str]:
    """
    Validate that current features match expected features.
    
    Args:
        current_features: Current feature list
        expected_features: Expected feature list
    
    Returns:
        (is_valid, message) tuple
    """
    current_set = set(current_features)
    expected_set = set(expected_features)
    
    missing = expected_set - current_set
    extra = current_set - expected_set
    
    if missing or extra:
        msg = []
        if missing:
            msg.append(f"Missing features: {sorted(missing)}")
        if extra:
            msg.append(f"Extra features: {sorted(extra)}")
        return False, "; ".join(msg)
    
    # Check order (important for some models)
    if current_features != expected_features:
        logger.warning("Feature order differs from expected")
    
    return True, "Feature consistency validated"


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Comprehensive data quality report.
    
    Args:
        df: DataFrame to check
    
    Returns:
        Dictionary with quality metrics
    """
    report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'n_duplicates': df.duplicated().sum(),
        'missing_by_column': df.isnull().sum().to_dict(),
        'total_missing_pct': df.isnull().sum().sum() / (len(df) * len(df.columns)),
        'dtypes': df.dtypes.astype(str).to_dict(),
    }
    
    # Numeric column stats
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numeric_cols) > 0:
        report['numeric_stats'] = df[numeric_cols].describe().to_dict()
    
    # Categorical column stats
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        report['categorical_unique_counts'] = {
            col: df[col].nunique() for col in cat_cols
        }
    
    return report
