# btc-quant-prob/models/lgbm_quantile.py

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

class LightGBMQuantile:
    def __init__(self, quantiles, lgbm_params):
        self.quantiles = quantiles
        self.lgbm_params = lgbm_params
        self.models = {}

    def fit(self, X_train, y_train):
        print("Training LightGBM Quantile models...")
        for q in tqdm(self.quantiles, desc="Training Quantiles"):
            params = self.lgbm_params.copy()
            params['alpha'] = q
            
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            self.models[q] = model
        return self

    def predict(self, X_test):
        predictions = pd.DataFrame(index=X_test.index)
        for q, model in self.models.items():
            predictions[f'q_{q}'] = model.predict(X_test)
        
        # Ensure quantiles are monotonically increasing
        predictions = predictions.cummax(axis=1)
        return predictions