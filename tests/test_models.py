# btc-quant-prob/tests/test_models.py

import pytest
import pandas as pd
import numpy as np
from models.lgbm_quantile import LightGBMQuantile

@pytest.fixture
def sample_train_data():
    """Creates sample training data."""
    X = pd.DataFrame(np.random.rand(100, 10), columns=[f'f{i}' for i in range(10)])
    y = pd.Series(np.random.rand(100))
    return X, y

def test_lgbm_quantile_predict_shape(sample_train_data):
    """Tests that the LightGBM quantile model returns predictions of the correct shape."""
    X, y = sample_train_data
    quantiles = [0.1, 0.5, 0.9]
    params = {'n_estimators': 10}
    
    model = LightGBMQuantile(quantiles, params)
    model.fit(X, y)
    
    X_test = pd.DataFrame(np.random.rand(20, 10), columns=X.columns)
    predictions = model.predict(X_test)
    
    assert predictions.shape == (20, len(quantiles))
    assert all([f'q_{q}' in predictions.columns for q in quantiles])