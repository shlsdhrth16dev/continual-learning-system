"""
Unit tests for the model registry.
Tests thread-safe operations, validation, and lifecycle management.
"""
import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from src.registry.model_registry import (
    initialize_registry, register_model, promote_model,
    get_model_by_version, get_production_model, list_all_models,
    load_registry, save_registry, rollback_to_version,
    archive_old_models, get_model_metrics_comparison,
    RegistryError
)


@pytest.fixture
def temp_registry_dir(monkeypatch):
    """Create a temporary directory for registry during tests."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Monkey patch the registry path
    from src.utils import paths
    monkeypatch.setattr(paths, 'REGISTRY_PATH', temp_path / 'registry.json')
    monkeypatch.setattr(paths, 'REGISTRY_DIR', temp_path)
    monkeypatch.setattr(paths, 'MODELS_DIR', temp_path / 'models')
    
    yield temp_path
    
    # Cleanup
    shutil.rmtree(temp_dir)


class TestRegistryInitialization:
    """Test registry initialization."""
    
    def test_initialize_new_registry(self, temp_registry_dir):
        """Test creating a new registry."""
        from src.utils.paths import REGISTRY_PATH
        
        initialize_registry()
        
        assert REGISTRY_PATH.exists()
        registry = load_registry()
        assert registry['latest_production'] is None
        assert registry['history'] == []
        assert 'created_at' in registry
    
    def test_initialize_existing_registry_fails(self, temp_registry_dir):
        """Test that initializing existing registry fails without force."""
        initialize_registry()
        
        with pytest.raises(RegistryError):
            initialize_registry(force=False)
    
    def test_initialize_with_force_overwrites(self, temp_registry_dir):
        """Test force initialization overwrites existing registry."""
        initialize_registry()
        
        # Add a model
        register_model({"accuracy": 0.8})
        
        # Force reinitialize
        initialize_registry(force=True)
        
        registry = load_registry()
        assert len(registry['history']) == 0


class TestModelRegistration:
    """Test model registration operations."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self, temp_registry_dir):
        """Initialize registry before each test."""
        initialize_registry()
    
    def test_register_first_model(self):
        """Test registering the first model."""
        metrics = {"accuracy": 0.85, "f1": 0.83}
        version = register_model(metrics)
        
        assert version == "model_v1"
        
        model = get_model_by_version(version)
        assert model['version'] == version
        assert model['metrics'] == metrics
        assert model['status'] == 'candidate'
    
    def test_register_multiple_models(self):
        """Test registering multiple models."""
        v1 = register_model({"accuracy": 0.85})
        v2 = register_model({"accuracy": 0.87})
        v3 = register_model({"accuracy": 0.90})
        
        assert v1 == "model_v1"
        assert v2 == "model_v2"
        assert v3 == "model_v3"
        
        all_models = list_all_models()
        assert len(all_models) == 3
    
    def test_register_with_metadata(self):
        """Test registering model with metadata."""
        metrics = {"accuracy": 0.85}
        metadata = {
            "data_hash": "abc123",
            "n_train": 1000,
            "feature_names": ["f1", "f2", "f3"]
        }
        config = {"n_estimators": 100}
        
        version = register_model(metrics, metadata, config)
        model = get_model_by_version(version)
        
        assert model['metadata'] == metadata
        assert model['config'] == config


class TestModelPromotion:
    """Test model promotion and lifecycle management."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self, temp_registry_dir):
        """Initialize registry with some models."""
        initialize_registry()
        self.v1 = register_model({"accuracy": 0.85})
        self.v2 = register_model({"accuracy": 0.90})
    
    def test_promote_model_to_production(self):
        """Test promoting a model to production."""
        promote_model(self.v1)
        
        prod_model = get_production_model()
        assert prod_model['version'] == self.v1
        assert prod_model['status'] == 'production'
        
        registry = load_registry()
        assert registry['latest_production'] == self.v1
    
    def test_promote_second_model_archives_first(self):
        """Test that promoting a new model archives the old one."""
        promote_model(self.v1)
        promote_model(self.v2)
        
        # Check v2 is production
        prod_model = get_production_model()
        assert prod_model['version'] == self.v2
        
        # Check v1 is archived
        v1_model = get_model_by_version(self.v1)
        assert v1_model['status'] == 'archived'
    
    def test_promote_nonexistent_model_fails(self):
        """Test promoting non-existent model raises error."""
        with pytest.raises(RegistryError):
            promote_model("model_v999")
    
    def test_promote_with_reason(self):
        """Test promotion with reason is recorded."""
        reason = "Significant F1 improvement"
        promote_model(self.v1, reason=reason)
        
        model = get_model_by_version(self.v1)
        assert model['promotion_reason'] == reason


class TestRollback:
    """Test rollback functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self, temp_registry_dir):
        """Initialize registry with promoted model."""
        initialize_registry()
        self.v1 = register_model({"accuracy": 0.85})
        self.v2 = register_model({"accuracy": 0.90})
        promote_model(self.v1)
        promote_model(self.v2)
    
    def test_rollback_to_previous_version(self):
        """Test rolling back to previous production version."""
        rollback_to_version(self.v1)
        
        prod_model = get_production_model()
        assert prod_model['version'] == self.v1
        assert prod_model['status'] == 'production'
        
        # v2 should be archived
        v2_model = get_model_by_version(self.v2)
        assert v2_model['status'] == 'archived'


class TestModelQueries:
    """Test model query functions."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self, temp_registry_dir):
        """Initialize registry with various models."""
        initialize_registry()
        self.v1 = register_model({"accuracy": 0.85})
        self.v2 = register_model({"accuracy": 0.87})
        self.v3 = register_model({"accuracy": 0.90})
        promote_model(self.v2)
    
    def test_list_all_models(self):
        """Test listing all models."""
        all_models = list_all_models()
        assert len(all_models) == 3
    
    def test_list_models_by_status(self):
        """Test filtering models by status."""
        candidates = list_all_models(status_filter='candidate')
        production = list_all_models(status_filter='production')
        
        assert len(candidates) == 2
        assert len(production) == 1
        assert production[0]['version'] == self.v2
    
    def test_get_production_model_when_none(self, temp_registry_dir):
        """Test getting production model when none exists."""
        initialize_registry()
        
        prod_model = get_production_model()
        assert prod_model is None
    
    def test_get_model_by_version_not_found(self):
        """Test getting non-existent version returns None."""
        model = get_model_by_version("model_v999")
        assert model is None


class TestArchiving:
    """Test model archiving functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self, temp_registry_dir):
        """Initialize registry with many candidate models."""
        initialize_registry()
        self.versions = []
        for i in range(10):
            version = register_model({"accuracy": 0.8 + i * 0.01})
            self.versions.append(version)
    
    def test_archive_old_candidates(self):
        """Test archiving old candidate models."""
        archived_count = archive_old_models(keep_recent=3)
        
        assert archived_count == 7  # 10 - 3 = 7
        
        candidates = list_all_models(status_filter='candidate')
        archived = list_all_models(status_filter='archived')
        
        assert len(candidates) == 3
        assert len(archived) == 7


class TestMetricsComparison:
    """Test metrics comparison functionality."""
    
    @pytest.fixture(autouse=True)
    def setup_registry(self, temp_registry_dir):
        """Initialize registry with two models."""
        initialize_registry()
        self.v1 = register_model({"accuracy": 0.85, "f1": 0.83})
        self.v2 = register_model({"accuracy": 0.90, "f1": 0.88})
    
    def test_compare_metrics_between_models(self):
        """Test comparing metrics between two models."""
        comparison = get_model_metrics_comparison(self.v1, self.v2)
        
        assert comparison['version1'] == self.v1
        assert comparison['version2'] == self.v2
        assert 'differences' in comparison
        
        # Check accuracy difference
        acc_diff = comparison['differences']['accuracy']
        assert acc_diff['absolute'] == pytest.approx(0.05)
        assert acc_diff['relative'] == pytest.approx(5.88, rel=0.1)
    
    def test_compare_nonexistent_models_fails(self):
        """Test comparing non-existent models raises error."""
        with pytest.raises(RegistryError):
            get_model_metrics_comparison(self.v1, "model_v999")


class TestRegistryValidation:
    """Test registry schema validation."""
    
    def test_load_invalid_json_fails(self, temp_registry_dir):
        """Test loading invalid JSON fails."""
        from src.utils.paths import REGISTRY_PATH
        
        # Create invalid JSON
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, 'w') as f:
            f.write("invalid json{")
        
        with pytest.raises(RegistryError, match="Invalid JSON"):
            load_registry()
    
    def test_load_missing_keys_fails(self, temp_registry_dir):
        """Test loading registry with missing keys fails."""
        from src.utils.paths import REGISTRY_PATH
        
        # Create registry with missing keys
        REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(REGISTRY_PATH, 'w') as f:
            json.dump({"latest_production": None}, f)
        
        with pytest.raises(RegistryError, match="missing required key"):
            load_registry()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
