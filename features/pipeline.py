# btc-quant-prob/features/pipeline.py

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# Custom transformers could be added here if needed

def create_feature_pipeline():
    """
    Creates a scikit-learn pipeline for feature preprocessing.
    This ensures that data transformations are applied consistently.
    """
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')), # Handles any potential NaNs from rolling features
        ('scaler', StandardScaler())
    ])
    return pipeline