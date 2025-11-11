# btc-quant-prob/models/ensemble.py

import pandas as pd

class SimpleEnsemble:
    def __init__(self, models: dict):
        """
        Args:
            models (dict): A dictionary of fitted model instances.
        """
        self.models = models

    def predict(self, X_test):
        """
        Averages the quantile predictions from all base models.
        A more advanced version would use weighted averaging based on validation performance.
        """
        print("Predicting with Simple Ensemble (Averaging)...")
        all_predictions = []
        for name, model in self.models.items():
            print(f"Getting predictions from {name}...")
            preds = model.predict(X_test)
            all_predictions.append(preds)
        
        # Average the predictions
        ensemble_preds = pd.concat(all_predictions).groupby(level=0).mean()
        return ensemble_preds