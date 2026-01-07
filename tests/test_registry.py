import os
import shutil
from pathlib import Path
import pytest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from src.registry.model_registry import register_model, load_model, get_latest_version, REGISTRY_ROOT


# Override REGISTRY_ROOT for tests
import src.registry.model_registry as mr

@pytest.fixture(autouse=True)
def mock_registry(tmp_path):
    """Ensures each test uses a temporary registry."""
    original_root = mr.REGISTRY_ROOT
    mr.REGISTRY_ROOT = tmp_path / "test_registry"
    yield
    mr.REGISTRY_ROOT = original_root


def test_register_and_load_model():
    model = LogisticRegression()
    version = "v1"
    metadata = {"accuracy": 0.85, "features": ["a", "b"]}
    
    register_model(model, metadata, version)
    
    # Verify files exist
    assert (REGISTRY_ROOT / "model_v1" / "model.pkl").exists()
    assert (REGISTRY_ROOT / "model_v1" / "metadata.json").exists()
    
    # Load back
    loaded_model, loaded_meta = load_model(version)
    assert loaded_meta["accuracy"] == 0.85
    assert isinstance(loaded_model, LogisticRegression)

def test_get_latest_version():
    model = LogisticRegression()
    register_model(model, {}, "v1")
    register_model(model, {}, "v2")
    register_model(model, {}, "v10")
    
    latest = get_latest_version()
    assert latest == "v10"

def test_load_non_existent_version():
    with pytest.raises(FileNotFoundError):
        load_model("v999")
