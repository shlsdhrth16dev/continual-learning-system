"""
Model training module with comprehensive configuration and validation.
"""
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.config import get_config
from src.utils.paths import get_model_version_path, ensure_directory
from src.utils.logging_config import get_logger
from src.utils.validation import (
    validate_dataframe_schema,
    validate_target_distribution,
    compute_data_hash,
    ValidationError
)

logger = get_logger(__name__)


def calculate_class_weight(y_train: pd.Series) -> str:
    """
    Calculate appropriate class_weight parameter for imbalanced data.
    
    Args:
        y_train: Training target series
    
    Returns:
        Class weight parameter ("balanced" or None)
    """
    value_counts = y_train.value_counts()
    if len(value_counts) < 2:
        return None
    
    max_count = value_counts.max()
    min_count = value_counts.min()
    imbalance_ratio = max_count / min_count
    
    if imbalance_ratio > 3:
        logger.info(f"Class imbalance detected ({imbalance_ratio:.1f}:1). Using balanced class weights.")
        return "balanced"
    
    return None


def split_data(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split data into train/validation/test sets with stratification.
    
    Args:
        df: Full dataset
        target_col: Target column name
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    config = get_config()
    
    if target_col not in df.columns:
        raise ValidationError(f"Target column '{target_col}' not found in data")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Validate target
    is_valid, msg = validate_target_distribution(y)
    if not is_valid:
        raise ValidationError(msg)
    
    # First split: train+val vs test
    test_size = config.data.test_ratio
    train_val_size = 1 - test_size
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=config.data.random_state,
        stratify=y if len(y.unique()) > 1 else None
    )
    
    # Second split: train vs val
    val_relative_size = config.data.val_ratio / train_val_size
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_relative_size,
        random_state=config.data.random_state,
        stratify=y_train_val if len(y_train_val.unique()) > 1 else None
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_model(
    data_path: Path,
    model_version: str,
    target_col: str = "target"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Train a model with proper data splitting and validation.
    
    Args:
        data_path: Path to training data CSV
        model_version: Version identifier for this model
        target_col: Name of target column
    
    Returns:
        Tuple of (trained model, training metadata)
    
    Raises:
        ValidationError: If data validation fails
    """
    config = get_config()
    logger.info(f"Training model {model_version} from {data_path}")
    
    # Load data
    df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Validate schema
    is_valid, msg = validate_dataframe_schema(df, min_rows=50)
    if not is_valid:
        raise ValidationError(f"Data validation failed: {msg}")
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_col)
    
    # Calculate class weights
    class_weight = calculate_class_weight(y_train)
    
    # Get model config and override class_weight if needed
    model_config = config.model.to_dict()
    if class_weight:
        model_config['class_weight'] = class_weight
    
    logger.info(f"Training with config: {model_config}")
    
    # Train model
    model = RandomForestClassifier(**model_config)
    model.fit(X_train, y_train)
    
    # Save model and splits
    model_dir = get_model_version_path(model_version)
    ensure_directory(model_dir)
    
    joblib.dump(model, model_dir / "model.pkl")
    logger.info(f"Model saved to {model_dir / 'model.pkl'}")
    
    # Save test set for evaluation
    X_test.to_csv(model_dir / "X_test.csv", index=False)
    y_test.to_csv(model_dir / "y_test.csv", index=False)
    
    # Create metadata
    metadata = {
        "data_hash": compute_data_hash(df),
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "feature_names": X_train.columns.tolist(),
        "target_distribution_train": y_train.value_counts().to_dict(),
        "class_weight_used": class_weight,
        "training_config": model_config
    }
    
    logger.info(f"Training complete for {model_version}")
    
    return model, metadata

