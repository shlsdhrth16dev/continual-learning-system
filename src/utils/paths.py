"""
Path management for the continual learning system.
Provides centralized path definitions and utilities.
"""
from pathlib import Path
from typing import List

# Base directory
BASE_DIR = Path(__file__).resolve().parents[2]

# Main directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
SCRIPTS_DIR = BASE_DIR / "scripts"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
INFERENCE_DATA_DIR = DATA_DIR / "inference"
REFERENCE_STATS_DIR = DATA_DIR / "reference"
DRIFT_REPORTS_DIR = DATA_DIR / "drift_reports"

# Model registry
REGISTRY_DIR = MODELS_DIR / "registry"
REGISTRY_PATH = REGISTRY_DIR / "registry.json"

# Preprocessor
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"


def ensure_directory(path: Path) -> Path:
    """
    Ensure a directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        The path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_all_directories() -> None:
    """Create all required directories."""
    directories = [
        DATA_DIR,
        MODELS_DIR,
        LOGS_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        INFERENCE_DATA_DIR,
        REFERENCE_STATS_DIR,
        DRIFT_REPORTS_DIR,
        REGISTRY_DIR,
    ]
    
    for directory in directories:
        ensure_directory(directory)


def validate_path_exists(path: Path, path_type: str = "file") -> None:
    """
    Validate that a path exists.
    
    Args:
        path: Path to check
        path_type: Type of path ("file" or "directory")
    
    Raises:
        FileNotFoundError: If path doesn't exist
    """
    if not path.exists():
        raise FileNotFoundError(f"{path_type.capitalize()} not found: {path}")
    
    if path_type == "file" and not path.is_file():
        raise FileNotFoundError(f"Expected file but found directory: {path}")
    
    if path_type == "directory" and not path.is_dir():
        raise FileNotFoundError(f"Expected directory but found file: {path}")


def get_model_version_path(version: str) -> Path:
    """
    Get the directory path for a specific model version.
    
    Args:
        version: Model version identifier (e.g., "model_v1")
    
    Returns:
        Path to model version directory
    """
    return MODELS_DIR / version


def get_latest_file(directory: Path, pattern: str = "*") -> Path:
    """
    Get the most recently modified file in a directory.
    
    Args:
        directory: Directory to search
        pattern: File pattern (e.g., "*.csv")
    
    Returns:
        Path to latest file
    
    Raises:
        FileNotFoundError: If no files match pattern
    """
    validate_path_exists(directory, "directory")
    
    files = list(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {directory}")
    
    return max(files, key=lambda p: p.stat().st_mtime)
