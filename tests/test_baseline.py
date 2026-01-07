import os
import joblib
import pandas as pd
import pytest
from src.registry.model_registry import load_model, get_latest_version


def test_latest_model_inference():
    """Ensures the latest model and preprocessor can be loaded and perform inference."""
    latest_v = get_latest_version()
    if not latest_v:
        pytest.skip("No model versions found.")
    
    model, metadata = load_model(latest_v)
    
    # Load preprocessor
    preprocessor_path = "models/preprocessor.joblib"
    assert os.path.exists(preprocessor_path), "Preprocessor missing!"
    preprocessor = joblib.load(preprocessor_path)
    
    # Create mock raw input based on metadata feature list
    raw_features = metadata.get("features", [])
    if not raw_features:
        pytest.skip("Metadata missing feature list.")
        
    # Mock one raw row
    mock_raw = pd.DataFrame([[0.0] * len(raw_features)], columns=raw_features)
    
    # Transform
    mock_processed = preprocessor.transform(mock_raw)
    
    # Predict
    prediction = model.predict(mock_processed)
    assert len(prediction) == 1
    assert prediction[0] in [0, 1]


def test_reference_stats_file_exists():
    latest_v = get_latest_version()
    if not latest_v:
        pytest.skip("No model versions.")
        
    stats_path = f"data/reference/feature_stats_{latest_v}.json"
    assert os.path.exists(stats_path), f"Stats file {stats_path} missing!"
