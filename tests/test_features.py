# btc-quant-prob/tests/test_features.py

import pytest
import pandas as pd
import numpy as np
from features.build_features import build_feature_set

@pytest.fixture
def sample_ohlcv_data():
    """Creates a sample OHLCV DataFrame for testing."""
    dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=200))
    data = {
        'Open': np.random.rand(200) * 100 + 1000,
        'High': np.random.rand(200) * 100 + 1050,
        'Low': np.random.rand(200) * 100 + 950,
        'Close': np.random.rand(200) * 100 + 1000,
        'Volume_(BTC)': np.random.rand(200) * 10 + 1,
    }
    return pd.DataFrame(data, index=dates)

def test_build_feature_set_no_nan(sample_ohlcv_data):
    """Tests that the feature building process produces no NaN values in the output."""
    config = {
        "log_return_lags": [1, 7, 30],
        "volatility_windows": [30, 90],
        "rolling_stat_windows": [30, 90]
    }
    features = build_feature_set(sample_ohlcv_data, config)
    
    # After dropping initial NaNs, there should be none left
    assert not features.isnull().values.any()

def test_build_feature_set_correct_shape(sample_ohlcv_data):
    """Tests that the feature matrix has a reasonable shape."""
    config = {
        "log_return_lags": [1, 7, 30],
        "volatility_windows": [30, 90],
        "rolling_stat_windows": [30, 90]
    }
    features = build_feature_set(sample_ohlcv_data, config)
    
    # Original length - max window size for NaNs
    assert features.shape[0] < len(sample_ohlcv_data)
    assert features.shape[1] > 5 # Should have many more columns than inputs