# btc-quant-prob/backtest/walkforward.py

import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import argparse
from tqdm import tqdm

# Tambahkan import ini untuk path-finding
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.ingest import load_and_clean_data
from data.preprocess import create_target
from features.build_features import build_feature_set
from features.pipeline import create_feature_pipeline

from models.lgbm_quantile import LightGBMQuantile
from models.gp_baseline import GPBaseline
from models.ensemble import SimpleEnsemble

def run_walk_forward(
    config_path: str, 
    horizon: int, 
    model_name: str, 
    model_params_override: dict = None,
    show_progress: bool = True
) -> pd.DataFrame:
    """
    Performs a walk-forward validation of a model.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    backtest_config = config['backtest']

    # --- 1. Load and Prepare Data ---
    if show_progress:
        print("Preparing data for backtest...")
    df_daily = load_and_clean_data(config['data']['raw_path'])
    features = build_feature_set(df_daily, config['features'])
    target = create_target(df_daily, horizon)
    
    # --- PERBAIKAN DI SINI ---
    # Kita gabungkan: fitur, target, DAN 'Close' (untuk regime filter)
    data = pd.concat([features, target, df_daily['Close']], axis=1)
    # --- AKHIR PERBAIKAN ---

    # Drop NaNs dari rolling features, target creation, dan alignment
    data.dropna(inplace=True)
    data.index = pd.to_datetime(data.index)

    # --- 2. Walk-Forward Loop ---
    start_date = pd.to_datetime(backtest_config['start_date'])
    end_date = data.index.max() - pd.to_timedelta(horizon, unit='d')
    
    train_window = pd.to_timedelta(backtest_config['train_window_days'], unit='d')
    step = pd.to_timedelta(backtest_config['step_days'], unit='d')
    
    current_date = start_date
    all_predictions = []

    total_steps = (end_date - start_date).days // step.days
    pbar = tqdm(total=total_steps, desc="Walk-Forward Backtest", disable=not show_progress)

    while current_date < end_date:
        train_start = current_date - train_window
        train_end = current_date
        test_start = current_date
        test_end = current_date + step
        
        train_mask = (data.index >= train_start) & (data.index < train_end)
        test_mask = (data.index >= test_start) & (data.index < test_end)
        
        if train_mask.sum() < 200 or test_mask.sum() == 0: # Pastikan data train cukup
            current_date += step
            pbar.update(1)
            continue

        # Pisahkan 'Close' dari fitur murni
        feature_cols = [col for col in data.columns if col not in [target.name, 'Close']]
        X_train = data.loc[train_mask, feature_cols]
        y_train = data.loc[train_mask, target.name]
        X_test = data.loc[test_mask, feature_cols]
        y_test = data.loc[test_mask, target.name]


        pipeline = create_feature_pipeline()
        X_train_s = pd.DataFrame(pipeline.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_s = pd.DataFrame(pipeline.transform(X_test), index=X_test.index, columns=X_test.columns)

        model_config = config['models'][model_name].copy()
        if model_params_override:
            model_config.update(model_params_override)
        
        if model_name == 'lgbm_quantile':
            model = LightGBMQuantile(config['training']['quantiles'], model_config).fit(X_train_s, y_train)
        elif model_name == 'gp_baseline':
            model = GPBaseline(config['training']['quantiles'], model_config).fit(X_train_s, y_train)
        elif model_name == 'ensemble':
            lgbm = LightGBMQuantile(config['training']['quantiles'], model_config).fit(X_train_s, y_train)
            gp = GPBaseline(config['training']['quantiles'], config['models']['gp_baseline']).fit(X_train_s, y_train)
            model = SimpleEnsemble({'lgbm': lgbm, 'gp': gp})
        else:
            raise ValueError(f"Model {model_name} not supported for backtesting.")

        preds = model.predict(X_test_s)
        
        # --- SEKARANG INI AKAN BERHASIL ---
        # 'data' sekarang memiliki 'Close', jadi kita bisa mengambilnya
        fold_results = pd.concat([y_test, preds, data.loc[test_mask, 'Close']], axis=1)
        # --- AKHIR PERBAIKAN ---

        all_predictions.append(fold_results)

        current_date += step
        pbar.update(1)
    
    pbar.close()
    
    if not all_predictions:
        print("Warning: No predictions were generated. Check backtest date ranges.")
        return pd.DataFrame()

    backtest_results = pd.concat(all_predictions)
    
    if show_progress:
        print(f"\nWalk-forward backtest complete. Returning results.")
        
    return backtest_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run walk-forward backtest.")
    parser.add_argument('--config', type=str, default='train/config.json', help='Path to config file.')
    parser.add_argument('--horizon', type=int, required=True, help='Forecast horizon in days.')
    parser.add_argument('--model_name', type=str, required=True, help='Model to backtest.')
    
    args = parser.parse_args()
    
    results_df = run_walk_forward(args.config, args.horizon, args.model_name)
    
    if not results_df.empty:
        artifacts_dir = Path(f"artifacts/backtest")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifacts_dir / f'{args.model_name}_h{args.horizon}_predictions.parquet'
        results_df.to_parquet(output_path)
        print(f"Results saved to {output_path}")