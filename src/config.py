"""
Centralized configuration management for the continual learning system.
Provides type-safe configuration with validation and environment overrides.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional


@dataclass
class ModelConfig:
    """Configuration for model training."""
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42
    class_weight: str = "balanced"  # or "balanced_subsample", or None
    n_jobs: int = -1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for sklearn."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'random_state': self.random_state,
            'class_weight': self.class_weight,
            'n_jobs': self.n_jobs
        }


@dataclass
class DataConfig:
    """Configuration for data processing."""
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    target_column: str = "target"
    
    def __post_init__(self):
        """Validate split ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if not abs(total - 1.0) < 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")


@dataclass
class DriftConfig:
    """Configuration for drift detection."""
    psi_threshold: float = 0.2
    z_score_threshold: float = 3.0
    
    # Severity levels
    moderate_psi_threshold: float = 0.25
    high_psi_threshold: float = 0.5
    
    # Minimum drifted features to trigger retraining
    min_drifted_features_moderate: int = 3
    min_drifted_features_high: int = 1
    
    # Monitoring settings
    enable_auto_retraining: bool = True
    max_model_age_days: int = 30


@dataclass
class RetrainingConfig:
    """Configuration for retraining decisions."""
    # Minimum improvement required to promote (absolute)
    min_improvement_threshold: float = 0.01
    
    # Maximum acceptable degradation (absolute)
    max_degradation_threshold: float = 0.01
    
    # Metrics to evaluate (first is primary)
    evaluation_metrics: list = field(default_factory=lambda: [
        'accuracy', 'f1', 'roc_auc', 'precision', 'recall'
    ])
    
    # Primary metric for promotion decision
    primary_metric: str = 'f1'
    
    # Require statistical significance
    require_significance: bool = True
    significance_level: float = 0.05
    
    # Notification settings
    notify_on_promotion: bool = True
    notify_on_rejection: bool = True
    notify_on_error: bool = True


@dataclass
class PathConfig:
    """Configuration for file paths."""
    base_dir: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    
    def __post_init__(self):
        """Define all paths relative to base_dir."""
        self.data_dir = self.base_dir / "data"
        self.models_dir = self.base_dir / "models"
        self.logs_dir = self.base_dir / "logs"
        
        # Data subdirectories
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.inference_data_dir = self.data_dir / "inference"
        self.reference_stats_dir = self.data_dir / "reference"
        self.drift_reports_dir = self.data_dir / "drift_reports"
        
        # Model registry
        self.registry_dir = self.models_dir / "registry"
        self.registry_path = self.registry_dir / "registry.json"
        
        # Preprocessor
        self.preprocessor_path = self.models_dir / "preprocessor.joblib"


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    enable_file_logging: bool = True
    log_file: str = "continual_learning.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    
    # JSON logging for production
    enable_json_logging: bool = False


@dataclass
class Config:
    """Master configuration object."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    drift: DriftConfig = field(default_factory=DriftConfig)
    retraining: RetrainingConfig = field(default_factory=RetrainingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Environment
    env: str = field(default_factory=lambda: os.getenv("ENV", "development"))
    
    def create_directories(self) -> None:
        """Create all necessary directories."""
        directories = [
            self.paths.data_dir,
            self.paths.models_dir,
            self.paths.logs_dir,
            self.paths.raw_data_dir,
            self.paths.processed_data_dir,
            self.paths.inference_data_dir,
            self.paths.reference_stats_dir,
            self.paths.drift_reports_dir,
            self.paths.registry_dir,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
        _config.create_directories()
    return _config


def reset_config() -> None:
    """Reset the global configuration (useful for testing)."""
    global _config
    _config = None
