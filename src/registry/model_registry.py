import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib

logger = logging.getLogger(__name__)

REGISTRY_ROOT = Path("models/registry")

def register_model(model: Any, metadata: Dict[str, Any], version: str):
    """Registers a model and its metadata in the versioned registry."""
    model_path = REGISTRY_ROOT / f"model_{version}"
    model_path.mkdir(parents=True, exist_ok=True)

    # Save model
    joblib.dump(model, model_path / "model.pkl")
    logger.info(f"Model saved to {model_path}/model.pkl")

    # Save metadata
    with open(model_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {model_path}/metadata.json")

def load_model(version: str) -> Tuple[Any, Dict[str, Any]]:
    """Loads a specific model version and its metadata."""
    model_path = REGISTRY_ROOT / f"model_{version}"
    if not model_path.exists():
        raise FileNotFoundError(f"Model version {version} not found in registry at {model_path}")

    model = joblib.load(model_path / "model.pkl")
    with open(model_path / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    logger.info(f"Loaded model version {version}")
    return model, metadata

def get_latest_version() -> Optional[str]:
    """Finds the latest model version based on directory naming."""
    if not REGISTRY_ROOT.exists():
        return None
    
    versions = []
    for d in REGISTRY_ROOT.iterdir():
        if d.is_dir() and d.name.startswith("model_v"):
            try:
                v_num = int(d.name.replace("model_v", ""))
                versions.append(v_num)
            except ValueError:
                continue
    
    if not versions:
        return None
    
    return f"v{max(versions)}"

