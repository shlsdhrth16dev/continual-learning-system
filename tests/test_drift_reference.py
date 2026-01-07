import pandas as pd
import pytest
from src.drift.reference_stats import compute_reference_stats

def test_compute_reference_stats_numeric():
    df = pd.DataFrame({
        'feat1': [10, 20, 30, 40],
        'feat2': [1.5, 2.5, 3.5, 4.5]
    })
    stats = compute_reference_stats(df)
    
    assert 'feat1' in stats
    assert stats['feat1']['type'] == 'numeric'
    assert 'mean' in stats['feat1']['stats']
    assert stats['feat1']['stats']['mean'] == 25.0
    assert stats['feat1']['stats']['count'] == 4

def test_compute_reference_stats_categorical():
    df = pd.DataFrame({
        'cat': ['A', 'A', 'B', 'C']
    })
    stats = compute_reference_stats(df)
    
    assert 'cat' in stats
    assert stats['cat']['type'] == 'categorical'
    assert stats['cat']['stats']['unique_count'] == 3
    assert stats['cat']['stats']['value_counts']['A'] == 2

def test_empty_dataframe():
    df = pd.DataFrame(columns=['col1'])
    # This might fail or return empty depending on implementation, 
    # but let's ensure it doesn't crash badly.
    stats = compute_reference_stats(df)
    assert isinstance(stats, dict)
