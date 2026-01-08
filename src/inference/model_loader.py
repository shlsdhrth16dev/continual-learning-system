"""
Model loading utilities with caching and fallback capabilities.
"""
import joblib
from pathlib import Path
from typing import Optional, Any, Tuple, Dict

from src.registry.model_registry import load_registry, get_model_by_version, get_production_model
from src.utils.paths import get_model_version_path
from src.utils.logging_config import get_logger
from src.utils.validation import validate_model_artifact

logger = get_logger(__name__)

# Simple cache for loaded models
_model_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}


def load_model_by_version(
    version: str,
    use_cache: bool = True
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load a specific model version with metadata.
    
    Args:
        version: Model version to load
        use_cache: Whether to use cached model if available
    
    Returns:
        Tuple of (model, model_entry_dict)
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If model is invalid
    """
    # Check cache
    if use_cache and version in _model_cache:
        logger.debug(f"Loading {version} from cache")
        return _model_cache[version]
    
    # Get model entry from registry
    model_entry = get_model_by_version(version)
    if not model_entry:
        raise ValueError(f"Model version {version} not found in registry")
    
    # Validate model file exists and is loadable
    model_path = get_model_version_path(version) / "model.pkl"
    is_valid, msg = validate_model_artifact(model_path)
    
    if not is_valid:
        raise ValueError(f"Model artifact validation failed: {msg}")
    
    # Load model
    try:
        model = joblib.load(model_path)
        logger.info(f"Loaded model {version} from {model_path}")
        
        # Cache it
        if use_cache:
            _model_cache[version] = (model, model_entry)
        
        return model, model_entry
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model {version}: {e}")


def load_production_model(use_cache: bool = True) -> Tuple[Any, Dict[str, Any]]:
    """
    Load the current production model with fallback logic.
    
    Args:
        use_cache: Whether to use cached model if available
    
    Returns:
        Tuple of (model, model_entry_dict)
    
    Raises:
        RuntimeError: If no production model exists or cannot be loaded
    """
    prod_model_info = get_production_model()
    
    if not prod_model_info:
        raise RuntimeError("No production model found in registry")
    
    version = prod_model_info['version']
    
    try:
        return load_model_by_version(version, use_cache=use_cache)
    except Exception as e:
        logger.error(f"Failed to load production model {version}: {e}")
        
        # Try to fallback to previous production model
        registry = load_registry()
        archived_production = [
            m for m in registry['history']
            if m.get('status') == 'archived' and 'promoted_at' in m
        ]
        
        if archived_production:
            # Get most recent archived production model
            archived_production.sort(key=lambda x: x.get('archived_at', ''), reverse=True)
            fallback_version = archived_production[0]['version']
            
            logger.warning(f"Attempting fallback to {fallback_version}")
            try:
                return load_model_by_version(fallback_version, use_cache=use_cache)
            except Exception as fallback_error:
                raise RuntimeError(f"Fallback also failed: {fallback_error}")
        
        raise RuntimeError(f"No fallback model available: {e}")


def clear_model_cache(version: Optional[str] = None) -> None:
    """
   Clear the model cache.
    
    Args:
        version: Specific version to clear, or None to clear all
    """
    if version:
        if version in _model_cache:
            del _model_cache[version]
            logger.debug(f"Cleared cache for {version}")
    else:
        _model_cache.clear()
        logger.debug("Cleared entire model cache")


def get_cached_versions() -> list:
    """Get list of cached model versions."""
    return list(_model_cache.keys())

