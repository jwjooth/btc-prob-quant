# btc-quant-prob/train/train.py

import pandas as pd # type: ignore
import json
import joblib
from pathlib import Path
import argparse

from data.ingest import load_and_clean_data
from data.preprocess import create_target, split_data
from features.build_features import build_feature_set
from features.pipeline import create_feature_pipeline

from models.lgbm_quantile import LightGBMQuantile
from models.bayesian_lstm import BayesianLSTM
from models.gp_baseline import GPBaseline

def run_training(config_path: str, horizon: int, model_name: str):
    """
    Main training script.
    
    Args:
        config_path (str): Path to the config.json file.
        horizon (int): Forecast horizon in days (e.g., 180 or 365).
        model_name (str): Name of the model to train ('all', 'lgbm_quantile', etc.).
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # --- 1. Load and Prepare Data ---
    df_daily = load_and_clean_data(config['data']['raw_path'])
    features = build_feature_set(df_daily, config['features'])
    target = create_target(df_daily, horizon)
    
    # Combine and drop NaNs
    data = pd.concat([features, target], axis=1).dropna()
    X = data.drop(columns=[target.name])
    y = data[target.name]

    # --- 2. Split and Preprocess ---
    X_train, X_test, y_train, y_test = split_data(X, y, config['training']['test_split_date'])
    
    feature_pipeline = create_feature_pipeline()
    X_train_scaled = pd.DataFrame(feature_pipeline.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(feature_pipeline.transform(X_test), index=X_test.index, columns=X_test.columns)
    
    # --- 3. Train Model(s) ---
    models_to_train = config['models'].keys() if model_name == 'all' else [model_name]
    
    for name in models_to_train:
        print(f"\n--- Training model: {name} for {horizon}d horizon ---")
        
        if name == 'lgbm_quantile':
            model = LightGBMQuantile(
                quantiles=config['training']['quantiles'],
                lgbm_params=config['models']['lgbm_quantile']
            )
        elif name == 'bayesian_lstm':
            # Note: This is a placeholder for the more complex LSTM training
            print("Skipping Bayesian LSTM training in this script due to complexity. See notebook.")
            continue
        elif name == 'gp_baseline':
            model = GPBaseline(
                quantiles=config['training']['quantiles'],
                gp_params=config['models']['gp_baseline']
            )
        else:
            print(f"Model {name} not recognized. Skipping.")
            continue

        model.fit(X_train_scaled, y_train)
        
        # --- 4. Save Artifacts ---
        artifacts_dir = Path(f"artifacts/models/{name}/h{horizon}")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, artifacts_dir / "model.joblib")
        joblib.dump(feature_pipeline, artifacts_dir / "pipeline.joblib")
        
        print(f"Model and pipeline for {name} (h{horizon}) saved to {artifacts_dir}")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train forecasting models.")
    parser.add_argument('--config', type=str, default='train/config.json', help='Path to config file.')
    parser.add_argument('--horizon', type=int, required=True, help='Forecast horizon in days (e.g., 180).')
    parser.add_argument('--model_name', type=str, default='all', help='Model to train (e.g., lgbm_quantile, all).')
    
    args = parser.parse_args()
    
    run_training(args.config, args.horizon, args.model_name)