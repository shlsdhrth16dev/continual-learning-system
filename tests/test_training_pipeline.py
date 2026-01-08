"""
Unit tests for the training and evaluation pipeline.
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

from src.training.trainer import train_model, split_data, calculate_class_weight
from src.training.evaluator import evaluate_model, calculate_metrics, mcnemar_test
from src.utils.validation import ValidationError


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    n_samples = 200
    
    data = {
        'feature1': np.random.randn(n_samples),
        'feature2': np.random.randn(n_samples),
        'feature3': np.random.randn(n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def temp_model_dir(monkeypatch):
    """Create temporary directory for models."""
    temp_dir = tempfile.mkdtemp()
    temp_path = Path(temp_dir)
    
    # Monkey patch paths
    from src.utils import paths
    monkeypatch.setattr(paths, 'MODELS_DIR', temp_path / 'models')
    
    # Initialize registry for tests
    registry_path = temp_path / 'models' / 'registry' / 'registry.json'
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    registry = {
        "latest_production": None,
        "history": []
    }
    with open(registry_path, 'w') as f:
        json.dump(registry, f)
    
    monkeypatch.setattr(paths, 'REGISTRY_PATH', registry_path)
    
    yield temp_path
    
    shutil.rmtree(temp_dir)


class TestDataSplitting:
    """Test data splitting functionality."""
    
    def test_split_data_balanced(self, sample_data):
        """Test splitting balanced data."""
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            sample_data, 'target'
        )
        
        # Check sizes
        total = len(sample_data)
        assert len(X_train) == pytest.approx(total * 0.7, abs=5)
        assert len(X_val) == pytest.approx(total * 0.15, abs=5)
        assert len(X_test) == pytest.approx(total * 0.15, abs=5)
        
        # Check no data leakage
        assert len(set(X_train.index) & set(X_test.index)) == 0
        assert len(set(X_val.index) & set(X_test.index)) == 0
    
    def test_split_data_missing_target_fails(self, sample_data):
        """Test splitting with missing target column fails."""
        with pytest.raises(ValidationError, match="not found"):
            split_data(sample_data, 'nonexistent_target')
    
    def test_split_preserves_features(self, sample_data):
        """Test that splits preserve feature columns."""
        X_train, X_val, X_test, _, _, _ = split_data(sample_data, 'target')
        
        expected_features = ['feature1', 'feature2', 'feature3']
        assert list(X_train.columns) == expected_features
        assert list(X_val.columns) == expected_features
        assert list(X_test.columns) == expected_features


class TestClassWeightCalculation:
    """Test class weight calculation."""
    
    def test_balanced_data_no_weights(self):
        """Test balanced data doesn't need weights."""
        y = pd.Series([0, 0, 1, 1, 0, 1])  # Balanced
        weight = calculate_class_weight(y)
        assert weight is None
    
    def test_imbalanced_data_gets_weights(self):
        """Test imbalanced data gets weights."""
        y = pd.Series([0] * 90 + [1] * 10)  # 9:1 imbalance
        weight = calculate_class_weight(y)
        assert weight == "balanced"
    
    def test_single_class_no_weights(self):
        """Test single class returns None."""
        y = pd.Series([1, 1, 1, 1])
        weight = calculate_class_weight(y)
        assert weight is None


class TestModelTraining:
    """Test model training pipeline."""
    
    def test_train_model_success(self, sample_data, temp_model_dir):
        """Test successful model training."""
        # Save sample data
        data_path = temp_model_dir / 'train_data.csv'
        sample_data.to_csv(data_path, index=False)
        
        # Train model
        model, metadata = train_model(data_path, "model_v1", target_col='target')
        
        # Check model is trained
        assert isinstance(model, RandomForestClassifier)
        assert hasattr(model, 'n_estimators')
        
        # Check metadata
        assert 'data_hash' in metadata
        assert 'n_train' in metadata
        assert 'n_val' in metadata
        assert 'n_test' in metadata
        assert 'feature_names' in metadata
        
        # Check splits sum to total
        total_samples = metadata['n_train'] + metadata['n_val'] + metadata['n_test']
        assert total_samples == len(sample_data)
    
    def test_train_model_saves_artifacts(self, sample_data, temp_model_dir):
        """Test that training saves model and test data."""
        from src.utils.paths import MODELS_DIR
        
        data_path = temp_model_dir / 'train_data.csv'
        sample_data.to_csv(data_path, index=False)
        
        model, metadata = train_model(data_path, "model_v1", target_col='target')
        
        model_dir = MODELS_DIR / "model_v1"
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "X_test.csv").exists()
        assert (model_dir / "y_test.csv").exists()
    
    def test_train_model_insufficient_data_fails(self, temp_model_dir):
        """Test training with insufficient data fails."""
        # Create tiny dataset
        tiny_data = pd.DataFrame({
            'f1': [1, 2, 3],
            'target': [0, 1, 0]
        })
        
        data_path = temp_model_dir / 'tiny_data.csv'
        tiny_data.to_csv(data_path, index=False)
        
        with pytest.raises(ValidationError, match="Insufficient data"):
            train_model(data_path, "model_v1", target_col='target')


class TestMetricsCalculation:
    """Test metrics calculation."""
    
    def test_calculate_basic_metrics(self):
        """Test calculating basic classification metrics."""
        y_true = pd.Series([0, 0, 1, 1, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check accuracy calculation
        # Matches: 6 out of 8 (indices 0, 2, 3, 4, 6, 7)
        expected_accuracy = 0.75
        assert metrics['accuracy'] == pytest.approx(expected_accuracy, abs=0.01)
    
    def test_calculate_metrics_with_probabilities(self):
        """Test metrics with probability predictions."""
        y_true = pd.Series([0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1])
        y_proba = np.array([
            [0.9, 0.1],
            [0.8, 0.2],
            [0.2, 0.8],
            [0.1, 0.9]
        ])
        
        metrics = calculate_metrics(y_true, y_pred, y_proba)
        
        assert 'roc_auc' in metrics
        assert 0 <= metrics['roc_auc'] <= 1


class TestMcNemarTest:
    """Test McNemar's statistical test."""
    
    def test_mcnemar_identical_predictions(self):
        """Test McNemar when predictions are identical."""
        y_true = pd.Series([0, 1, 0, 1, 0, 1])
        y_pred1 = np.array([0, 1, 0, 1, 0, 1])
        y_pred2 = np.array([0, 1, 0, 1, 0, 1])
        
        statistic, p_value = mcnemar_test(y_true, y_pred1, y_pred2)
        
        # Identical predictions should have high p-value
        assert p_value > 0.05
    
    def test_mcnemar_different_predictions(self):
        """Test McNemar with different predictions."""
        y_true = pd.Series([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred1 = np.array([0, 1, 0, 1, 0, 1, 0, 1])  # Perfect
        y_pred2 = np.array([1, 0, 1, 0, 1, 0, 1, 0])  # All wrong
        
        statistic, p_value = mcnemar_test(y_true, y_pred1, y_pred2)
        
        # Very different predictions should have low p-value
        assert p_value < 0.05


class TestModelEvaluation:
    """Test model evaluation pipeline."""
    
    def test_evaluate_model_on_test_set(self, sample_data, temp_model_dir):
        """Test evaluating model on test set."""
        from src.utils.paths import MODELS_DIR
        
        # Train a model first
        data_path = temp_model_dir / 'train_data.csv'
        sample_data.to_csv(data_path, index=False)
        
        model, metadata = train_model(data_path, "model_v1", target_col='target')
        
        # Evaluate
        results = evaluate_model(model, data_path, "model_v1", use_test_set=True)
        
        # Check results structure
        assert 'metrics' in results
        assert 'confusion_matrix' in results
        assert 'per_class_metrics' in results
        assert 'n_test_samples' in results
        
        # Check metrics present
        assert 'accuracy' in results['metrics']
        assert 'f1' in results['metrics']
    
    def test_evaluate_saves_results(self, sample_data, temp_model_dir):
        """Test that evaluation saves results."""
        from src.utils.paths import MODELS_DIR
        
        data_path = temp_model_dir / 'train_data.csv'
        sample_data.to_csv(data_path, index=False)
        
        model, metadata = train_model(data_path, "model_v1", target_col='target')
        results = evaluate_model(model, data_path, "model_v1", use_test_set=True)
        
        # Check evaluation file saved
        eval_path = MODELS_DIR / "model_v1" / "evaluation.json"
        assert eval_path.exists()
        
        # Load and verify
        import json
        with open(eval_path) as f:
            saved_results = json.load(f)
        
        assert 'metrics' in saved_results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
