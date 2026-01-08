"""
Simple integration tests to verify the system works end-to-end.
These tests are designed to run quickly and verify core functionality.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path


def test_config_module_importable():
    """Test that config module can be imported."""
    from src.config import get_config
    
    config = get_config()
    assert config is not None
    assert config.model.n_estimators > 0
    assert config.data.train_ratio > 0


def test_logging_setup():
    """Test that logging can be configured."""
    from src.utils.logging_config import get_logger
    
    logger = get_logger("test_logger")
    assert logger is not None
    logger.info("Test log message")


def test_validation_functions():
    """Test validation utilities."""
    from src.utils.validation import validate_dataframe_schema, compute_data_hash
    
    # Create test dataframe
    df = pd.DataFrame({
        'a': [1, 2, 3, 4, 5],
        'b': [6, 7, 8, 9, 10]
    })
    
    # Test schema validation
    is_valid, msg = validate_dataframe_schema(df, min_rows=3)
    assert is_valid
    
    # Test data hashing
    hash1 = compute_data_hash(df)
    assert isinstance(hash1, str)
    assert len(hash1) > 0
    
    # Same data should produce same hash
    hash2 = compute_data_hash(df)
    assert hash1 == hash2


def test_registry_basic_operations():
    """Test basic registry operations with temporary directory."""
    import sys
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    try:
        # Monkey patch the paths
        from src.utils import paths
        orig_registry_path = paths.REGISTRY_PATH
        orig_registry_dir = paths.REGISTRY_DIR
        orig_models_dir = paths.MODELS_DIR
        
        paths.REGISTRY_PATH = temp_path / 'registry' / 'registry.json'
        paths.REGISTRY_DIR = temp_path / 'registry'
        paths.MODELS_DIR = temp_path / 'models'
        
        # Now test registry
        from src.registry.model_registry import initialize_registry, register_model, load_registry
        
        initialize_registry()
        
        # Register a model
        version = register_model({"accuracy": 0.85, "f1": 0.83})
        assert version == "model_v1"
        
        # Load registry and verify
        registry = load_registry()
        assert len(registry['history']) == 1
        assert registry['history'][0]['version'] == 'model_v1'
        
        # Restore original paths
        paths.REGISTRY_PATH = orig_registry_path
        paths.REGISTRY_DIR = orig_registry_dir
        paths.MODELS_DIR = orig_models_dir
        
    finally:
        shutil.rmtree(temp_dir)


def test_data_splitting():
    """Test data splitting functionality."""
    from src.training.trainer import split_data
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'f1': np.random.randn(100),
        'f2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100)
    })
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, 'target')
    
    # Check splits are reasonable
    assert len(X_train) + len(X_val) + len(X_test) == 100
    assert len(X_train) > len(X_test)
    assert len(y_train) == len(X_train)


def test_metrics_calculation():
    """Test metrics calculation."""
    from src.training.evaluator import calculate_metrics
    
    y_true = pd.Series([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1' in metrics
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['f1'] <= 1


def test_class_weight_calculation():
    """Test class weight calculation."""
    from src.training.trainer import calculate_class_weight
    
    # Balanced data
    y_balanced = pd.Series([0, 0, 1, 1, 0, 1])
    weight = calculate_class_weight(y_balanced)
    assert weight is None
    
    # Imbalanced data
    y_imbalanced = pd.Series([0] * 90 + [1] * 10)
    weight = calculate_class_weight(y_imbalanced)
    assert weight == "balanced"


def test_paths_module():
    """Test paths module."""
    from src.utils.paths import BASE_DIR, DATA_DIR, MODELS_DIR, ensure_directory
    
    assert BASE_DIR.exists()
    assert isinstance(DATA_DIR, Path)
    assert isinstance(MODELS_DIR, Path)
    
    # Test ensure_directory
    temp_dir = tempfile.mkdtemp()
    test_path = Path(temp_dir) / "test" / "nested"
    ensure_directory(test_path)
    assert test_path.exists()
    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
