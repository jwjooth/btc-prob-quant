# btc-quant-prob/deploy/api.py

from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel
from pathlib import Path

app = FastAPI(title="Bitcoin Probabilistic Forecasting API")

class PredictionRequest(BaseModel):
    horizon_days: int

def load_latest_artifacts(model_name: str, horizon: int):
    """Helper to load the latest trained model and pipeline."""
    artifacts_dir = Path(f"artifacts/models/{model_name}/h{horizon}")
    if not artifacts_dir.exists():
        return None, None
        
    model = joblib.load(artifacts_dir / "model.joblib")
    pipeline = joblib.load(artifacts_dir / "pipeline.joblib")
    return model, pipeline

@app.post("/predict/")
def predict(request: PredictionRequest):
    """
    Serves the latest probabilistic forecast.
    Note: This endpoint needs live feature data to make a real prediction.
    Here, we mock it by using the last available data point.
    """
    model_name = "lgbm_quantile" # Default model for the API
    model, pipeline = load_latest_artifacts(model_name, request.horizon_days)

    if model is None:
        return {"error": f"No trained model found for horizon {request.horizon_days}"}

    # --- Mocking live feature data ---
    # In a real system, you would fetch and compute the latest features here.
    try:
        df = pd.read_parquet("artifacts/data/processed_btc_data.parquet")
        # For simplicity, we create features from this historical data.
        from features.build_features import build_feature_set
        import json
        with open('train/config.json', 'r') as f: config = json.load(f)
        features = build_feature_set(df, config['features'])
        latest_features = features.iloc[[-1]] # Get the last row
    except Exception as e:
        return {"error": f"Could not load or process feature data: {e}"}

    # Scale features and predict
    features_scaled = pipeline.transform(latest_features)
    predictions = model.predict(features_scaled)
    
    response = {
        "timestamp": latest_features.index[0].isoformat(),
        "horizon_days": request.horizon_days,
        "predicted_log_return_quantiles": predictions.to_dict(orient='records')[0]
    }
    
    return response

@app.get("/")
def read_root():
    return {"message": "Welcome to the BTC Forecasting API. POST to /predict/ to get forecasts."}