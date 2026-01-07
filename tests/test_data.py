
import pandas as pd
import pytest
from src.data.ingest import load_data
from src.data.preprocessor import preprocess

# Mocking data specifically for tests so we don't rely on local files
@pytest.fixture
def sample_csv(tmp_path):
    d = pd.DataFrame({
        'ID': [1, 2],
        'LIMIT_BAL': [20000, 120000],
        'SEX': [2, 2],
        'EDUCATION': [2, 2],
        'MARRIAGE': [1, 2],
        'AGE': [24, 26],
        'PAY_0': [2, -1],
        'default.payment.next.month': [1, 1]
    })
    p = tmp_path / "data.csv"
    d.to_csv(p, index=False)
    return str(p)

def test_load_data(sample_csv):
    df = load_data(sample_csv)
    assert not df.empty
    assert len(df) == 2

def test_preprocess(sample_csv):
    df = pd.read_csv(sample_csv)
    # Using the correct target column name as updated in preprocessor.py
    X, y = preprocess(df, target_col="default.payment.next.month", is_training=True)

    assert len(X) == len(y)
    assert X.shape[0] == 2
    assert X.isnull().sum().sum() == 0
