# btc-quant-prob/models/gp_baseline.py

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import pandas as pd
from scipy.stats import norm

class GPBaseline:
    def __init__(self, quantiles, gp_params):
        self.quantiles = quantiles
        self.params = gp_params
        kernel = 1.0 * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        self.model = GaussianProcessRegressor(
            kernel=kernel, 
            n_restarts_optimizer=self.params.get('n_restarts_optimizer', 10),
            random_state=42
        )

    def fit(self, X_train, y_train):
        print("Training Gaussian Process baseline...")
        # GP is computationally intensive, so we may need to subsample
        sample_size = min(2000, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)
        y_sample = y_train.loc[X_sample.index]
        self.model.fit(X_sample, y_sample)
        return self

    def predict(self, X_test):
        print("Predicting with Gaussian Process...")
        mean, std = self.model.predict(X_test, return_std=True)
        
        predictions = pd.DataFrame(index=X_test.index)
        for q in self.quantiles:
            predictions[f'q_{q}'] = norm.ppf(q, loc=mean, scale=std)
            
        return predictions