"""
Model registry for tracking and managing model versions.
Provides thread-safe operations for model lifecycle management.
"""
import json
try:
    import fcntl
    HAS_FCNTL = True
except ImportError:
    # Windows doesn't have fcntl
    HAS_FCNTL = False
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from contextlib import contextmanager

from src.utils.paths import REGISTRY_PATH, MODELS_DIR, ensure_directory
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


class RegistryError(Exception):
    """Raised when registry operations fail."""
    pass


@contextmanager
def _lock_registry():
    """
    Context manager for thread-safe registry operations.
    Uses file locking to prevent race conditions.
    """
    # Ensure registry directory exists
    ensure_directory(REGISTRY_PATH.parent)
    
    # Create lock file
    lock_file = REGISTRY_PATH.parent / ".registry.lock"
    
    try:
        with open(lock_file, 'w') as lock:
            # Acquire exclusive lock (if supported)
            if HAS_FCNTL:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                except OSError:
                    logger.warning("File locking failed, proceeding without lock")
            else:
                logger.debug("File locking not supported on this platform")
            
            yield
            
    finally:
        # Lock is automatically released when file is closed
        if lock_file.exists():
            try:
                lock_file.unlink()
            except Exception:
                pass


def _validate_registry_schema(registry: Dict[str, Any]) -> None:
    """
    Validate registry schema.
    
    Args:
        registry: Registry dictionary
    
    Raises:
        RegistryError: If schema is invalid
    """
    required_keys = ['latest_production', 'history']
    for key in required_keys:
        if key not in registry:
            raise RegistryError(f"Registry missing required key: {key}")
    
    if not isinstance(registry['history'], list):
        raise RegistryError("Registry 'history' must be a list")
    
    # Validate each model entry
    for i, model in enumerate(registry['history']):
        required_model_keys = ['version', 'trained_on', 'status', 'metrics']
        for key in required_model_keys:
            if key not in model:
                raise RegistryError(f"Model entry {i} missing required key: {key}")


def load_registry() -> Dict[str, Any]:
    """
    Load the model registry with validation.
    
    Returns:
        Registry dictionary
    
    Raises:
        RegistryError: If registry doesn't exist or is invalid
    """
    if not REGISTRY_PATH.exists():
        raise RegistryError(f"Registry not found at {REGISTRY_PATH}. Initialize registry first.")
    
    try:
        with open(REGISTRY_PATH, "r") as f:
            registry = json.load(f)
        
        _validate_registry_schema(registry)
        return registry
        
    except json.JSONDecodeError as e:
        raise RegistryError(f"Invalid JSON in registry: {e}")
    except Exception as e:
        raise RegistryError(f"Failed to load registry: {e}")


def save_registry(registry: Dict[str, Any]) -> None:
    """
    Save the model registry with validation.
    
    Args:
        registry: Registry dictionary
    
    Raises:
        RegistryError: If validation fails
    """
    _validate_registry_schema(registry)
    
    ensure_directory(REGISTRY_PATH.parent)
    
    try:
        # Write to temporary file first for atomicity
        temp_path = REGISTRY_PATH.parent / f"{REGISTRY_PATH.name}.tmp"
        with open(temp_path, "w") as f:
            json.dump(registry, f, indent=2)
        
        # Atomic rename
        temp_path.replace(REGISTRY_PATH)
        logger.debug(f"Registry saved to {REGISTRY_PATH}")
        
    except Exception as e:
        raise RegistryError(f"Failed to save registry: {e}")


def initialize_registry(force: bool = False) -> None:
    """
    Initialize an empty registry.
    
    Args:
        force: If True, overwrite existing registry
    
    Raises:
        RegistryError: If registry exists and force=False
    """
    if REGISTRY_PATH.exists() and not force:
        raise RegistryError(f"Registry already exists at {REGISTRY_PATH}")
    
    registry = {
        "latest_production": None,
        "history": [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    save_registry(registry)
    logger.info(f"Initialized registry at {REGISTRY_PATH}")


def get_next_version() -> str:
    """
    Get the next available version number.
    
    Returns:
        Version string (e.g., "model_v2")
    """
    registry = load_registry()
    
    if not registry["history"]:
        return "model_v1"
    
    versions = [
        int(m["version"].split("_v")[1])
        for m in registry["history"]
        if "_v" in m["version"]
    ]
    
    if not versions:
        return "model_v1"
    
    return f"model_v{max(versions) + 1}"


def register_model(
    metrics: Dict[str, float],
    metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Register a new model in the registry.
    
    Args:
        metrics: Model performance metrics
        metadata: Additional metadata (training data hash, features, etc.)
        config: Training configuration used
    
    Returns:
        Version string of registered model
    """
    with _lock_registry():
        registry = load_registry()
        new_version = get_next_version()
        
        model_entry = {
            "version": new_version,
            "trained_on": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": "candidate",
            "metrics": metrics,
            "metadata": metadata or {},
            "config": config or {},
            "created_at": datetime.now().isoformat()
        }
        
        registry["history"].append(model_entry)
        registry["updated_at"] = datetime.now().isoformat()
        save_registry(registry)
        
        logger.info(f"Registered model {new_version} with metrics: {metrics}")
        return new_version


def promote_model(version: str, reason: Optional[str] = None) -> None:
    """
    Promote a model to production status.
    
    Args:
        version: Version to promote
        reason: Optional reason for promotion
    
    Raises:
        RegistryError: If version doesn't exist or validation fails
    """
    with _lock_registry():
        registry = load_registry()
        
        # Find the model to promote
        model_found = False
        for model in registry["history"]:
            if model["version"] == version:
                model_found = True
                old_status = model["status"]
                model["status"] = "production"
                model["promoted_at"] = datetime.now().isoformat()
                if reason:
                    model["promotion_reason"] = reason
                registry["latest_production"] = version
                logger.info(f"Promoted {version} from {old_status} to production")
            elif model["status"] == "production":
                # Archive previous production model
                model["status"] = "archived"
                model["archived_at"] = datetime.now().isoformat()
                logger.info(f"Archived previous production model {model['version']}")
        
        if not model_found:
            raise RegistryError(f"Model version {version} not found in registry")
        
        registry["updated_at"] = datetime.now().isoformat()
        save_registry(registry)


def rollback_to_version(version: str) -> None:
    """
    Rollback to a previous model version.
    
    Args:
        version: Version to rollback to
    
    Raises:
        RegistryError: If version doesn't exist
    """
    logger.warning(f"Rolling back to version {version}")
    promote_model(version, reason="Rollback from failed deployment")


def get_model_by_version(version: str) -> Optional[Dict[str, Any]]:
    """
    Get model entry by version.
    
    Args:
        version: Model version
    
    Returns:
        Model entry dictionary or None if not found
    """
    registry = load_registry()
    
    for model in registry["history"]:
        if model["version"] == version:
            return model
    
    return None


def get_production_model() -> Optional[Dict[str, Any]]:
    """
    Get the current production model entry.
    
    Returns:
        Model entry dictionary or None if no production model
    """
    registry = load_registry()
    prod_version = registry.get("latest_production")
    
    if not prod_version:
        return None
    
    return get_model_by_version(prod_version)


def list_all_models(status_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    List all models in the registry.
    
    Args:
        status_filter: Optional status to filter by ('production', 'candidate', 'archived')
    
    Returns:
        List of model entries
    """
    registry = load_registry()
    
    if status_filter:
        return [m for m in registry["history"] if m["status"] == status_filter]
    
    return registry["history"]


def archive_old_models(keep_recent: int = 5) -> int:
    """
    Archive old candidate models, keeping only recent ones.
    
    Args:
        keep_recent: Number of recent candidates to keep
    
    Returns:
        Number of models archived
    """
    with _lock_registry():
        registry = load_registry()
        
        candidates = [m for m in registry["history"] if m["status"] == "candidate"]
        
        # Sort by creation date (newest first)
        candidates.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        archived_count = 0
        for model in candidates[keep_recent:]:
            model["status"] = "archived"
            model["archived_at"] = datetime.now().isoformat()
            model["archive_reason"] = "Automatic cleanup"
            archived_count += 1
            logger.info(f"Auto-archived old candidate {model['version']}")
        
        if archived_count > 0:
            registry["updated_at"] = datetime.now().isoformat()
            save_registry(registry)
        
        return archived_count


def get_model_metrics_comparison(version1: str, version2: str) -> Dict[str, Any]:
    """
    Compare metrics between two model versions.
    
    Args:
        version1: First version
        version2: Second version
    
    Returns:
        Dictionary with comparison results
    """
    model1 = get_model_by_version(version1)
    model2 = get_model_by_version(version2)
    
    if not model1 or not model2:
        raise RegistryError(f"One or both versions not found: {version1}, {version2}")
    
    comparison = {
        "version1": version1,
        "version2": version2,
        "metrics1": model1["metrics"],
        "metrics2": model2["metrics"],
        "differences": {}
    }
    
    # Calculate differences
    for metric in set(model1["metrics"].keys()) | set(model2["metrics"].keys()):
        val1 = model1["metrics"].get(metric, 0)
        val2 = model2["metrics"].get(metric, 0)
        comparison["differences"][metric] = {
            "absolute": val2 - val1,
            "relative": ((val2 - val1) / val1 * 100) if val1 != 0 else None
        }
    
    return comparison




