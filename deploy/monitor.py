# btc-quant-prob/deploy/monitor.py

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import subprocess
import time

# Add project root to sys.path to allow importing from other modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.prob_metrics import picp

def load_monitoring_data(predictions_path: Path):
    """
    Loads backtest predictions which will serve as the source of
    'live' predictions and actuals for the monitoring simulation.
    In a real system, this would query a production database.
    """
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions file not found at {predictions_path}. "
                                "Run a backtest first to generate it.")
    
    print(f"Loading historical prediction data from {predictions_path}...")
    df = pd.read_parquet(predictions_path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df

def run_monitoring_simulation(
    predictions_path: Path,
    config_path: Path,
    monitoring_window_days: int,
    check_interval_days: int,
    retrain_trigger_threshold: float
):
    """
    Simulates a monitoring loop over historical prediction data to detect model drift.

    Args:
        predictions_path (Path): Path to the backtest predictions file.
        config_path (Path): Path to the project config file.
        monitoring_window_days (int): The size of the rolling window for evaluation (e.g., 90 days).
        check_interval_days (int): How often to run the check (e.g., every 7 days).
        retrain_trigger_threshold (float): The tolerance for PICP drift (e.g., 0.05).
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    quantiles = config['training']['quantiles']
    nominal_coverage = quantiles[-1] - quantiles[0]
    
    df = load_monitoring_data(predictions_path)
    
    start_date = df.index.min() + pd.Timedelta(days=monitoring_window_days)
    end_date = df.index.max()
    current_date = start_date
    
    print("\n--- Starting Monitoring Simulation ---")
    print(f"Nominal Coverage (Target PICP): {nominal_coverage:.2f}")
    print(f"Drift Threshold: +/- {retrain_trigger_threshold:.2f}")
    print(f"Monitoring Window: {monitoring_window_days} days")
    print("--------------------------------------\n")
    
    while current_date <= end_date:
        print(f"[{current_date.date()}] Running monitoring check...")
        
        # Define the rolling window for this check
        window_start = current_date - pd.Timedelta(days=monitoring_window_days)
        window_data = df.loc[window_start:current_date]
        
        if len(window_data) < 30: # Ensure enough data for a meaningful check
            print("  -> Not enough data in window. Skipping.")
            current_date += pd.Timedelta(days=check_interval_days)
            continue
            
        y_true = window_data.iloc[:, 0].values
        lower_bound = window_data.iloc[:, 1].values # Assumes 1st quantile is lower bound
        upper_bound = window_data.iloc[:, -1].values # Assumes last quantile is upper bound
        
        # Calculate current PICP on the rolling window
        current_picp = picp(y_true, lower_bound, upper_bound)
        
        print(f"  -> Rolling PICP: {current_picp:.3f}")
        
        # Check for significant drift
        drift = abs(current_picp - nominal_coverage)
        if drift > retrain_trigger_threshold:
            print(f"  🚨 ALERT: Calibration drift detected! Drift = {drift:.3f} > {retrain_trigger_threshold:.3f}")
            print("  -> TRIGGERING MODEL RETRAINING...")
            
            # In a real system, this would trigger a CI/CD pipeline or a job scheduler.
            # Here, we simulate it by preparing and printing the training command.
            try:
                # Extract model and horizon from filename, e.g., 'lgbm_quantile_h180_predictions.parquet'
                parts = predictions_path.stem.split('_')
                model_name = '_'.join(parts[:-2])
                horizon = int(parts[-2].replace('h', ''))
                
                retrain_command = f"python -m train.train --horizon {horizon} --model_name {model_name}"
                print(f"  -> (Simulation) Would execute: `{retrain_command}`")
                
                # Uncomment the line below to actually run the retraining script
                # subprocess.run(retrain_command.split(), check=True, capture_output=True, text=True)

                print("  -> Retraining process would update the model artifacts for the API.")
                
            except Exception as e:
                print(f"  -> ERROR: Failed to prepare retraining command: {e}")
            
        else:
            print("  -> Model calibration is within tolerance. No action needed.")
        
        print("-" * 35)
        current_date += pd.Timedelta(days=check_interval_days)
        time.sleep(0.1) # small sleep to make simulation output readable

    print("\n--- Monitoring Simulation Complete ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run model performance monitoring simulation.")
    parser.add_argument(
        '--preds-file', 
        type=str, 
        default='artifacts/backtest/lgbm_quantile_h180_predictions.parquet',
        help='Path to the backtest predictions file to monitor.'
    )
    parser.add_argument('--config', type=str, default='train/config.json', help='Path to config file.')
    parser.add_argument('--window', type=int, default=90, help='Rolling window size in days for PICP calculation.')
    parser.add_argument('--interval', type=int, default=7, help='How often to run the check in days.')
    parser.add_argument('--threshold', type=float, default=0.05, help='PICP drift tolerance before triggering retrain.')
    
    args = parser.parse_args()
    
    run_monitoring_simulation(
        predictions_path=Path(args.preds_file),
        config_path=Path(args.config),
        monitoring_window_days=args.window,
        check_interval_days=args.interval,
        retrain_trigger_threshold=args.threshold
    )