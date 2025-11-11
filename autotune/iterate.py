# btc-quant-prob/autotune/iterate.py

import json
import pandas as pd
from pathlib import Path
import optuna
import joblib
import numpy as np # Pastikan numpy diimpor

# Tambahkan import ini untuk path-finding
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtest.walkforward import run_walk_forward
from backtest.simulate import simulate_trading, calculate_performance_metrics
from eval.prob_metrics import crps

# --- Konfigurasi Global untuk Tuning ---
CONFIG_PATH = 'train/config.json'
MODEL_TO_TUNE = 'lgbm_quantile'
HORIZON = 180
N_TRIALS = 50 # Jumlah eksperimen (Anda bisa naikkan jika mau)
# ----------------------------------------

def objective(trial: optuna.trial.Trial) -> float:
    """
    Fungsi 'objective' yang akan dioptimalkan oleh Optuna.
    Tujuan: Mencari SHARPE RATIO TERTINGGI.
    """
    
    # 1. Tentukan Hyperparameter Model (LGBM)
    model_params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'n_estimators': trial.suggest_int('n_estimators', 500, 2000, step=100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
    }

    # --- PERUBAHAN BARU ---
    # 2. Tentukan Hyperparameter Strategi (Parameter 'k')
    k_value = trial.suggest_float('k_value', 0.2, 2.5) # Cari k dari 0.2 (lambat) s/d 2.5 (agresif)
    # --- AKHIR PERUBAHAN ---

    print(f"\n--- Trial {trial.number}: Testing params {model_params} | k = {k_value:.2f} ---")

    try:
        # 3. Jalankan backtest penuh dengan parameter model
        results_df = run_walk_forward(
            config_path=CONFIG_PATH,
            horizon=HORIZON,
            model_name=MODEL_TO_TUNE,
            model_params_override=model_params,
            show_progress=False 
        )

        if results_df.empty:
            print(f"Trial {trial.number} failed: No predictions generated.")
            return -float('inf') # Penalti untuk MAKSIMASI

        # 4. Hitung Metrik Performa
        with open(CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # --- PERUBAHAN BARU ---
        # Kirim 'k_value' ke fungsi simulasi
        strategy_returns = simulate_trading(
            results_df, 
            config['backtest']['transaction_cost_pct'],
            k_value=k_value
        )
        # --- AKHIR PERUBAHAN ---

        performance = calculate_performance_metrics(strategy_returns)
        
        # Ambil Sharpe Ratio sebagai float, beri 0.0 jika gagal
        try:
            sharpe_ratio = float(performance.get('Sharpe Ratio', 0.0))
        except (ValueError, TypeError):
            sharpe_ratio = 0.0

        # Simpan metrik sekunder
        trial.set_user_attr('sharpe_ratio', sharpe_ratio)
        trial.set_user_attr('max_drawdown', performance.get('Max Drawdown', '-100.00%'))

        print(f"--- Trial {trial.number} Result: Sharpe = {sharpe_ratio:.2f} ---")

        # --- PERUBAHAN BARU ---
        # 5. Kembalikan metrik utama (yang ingin kita MAKSIMALKAN)
        return sharpe_ratio
        # --- AKHIR PERUBAHAN ---

    except Exception as e:
        print(f"Trial {trial.number} FAILED with exception: {e}")
        return -float('inf') # Penalti untuk MAKSIMASI


if __name__ == '__main__':
    print("--- Starting Optuna Hyperparameter Tuning (Goal: MAXIMIZE Sharpe Ratio) ---")
    print(f"Model: {MODEL_TO_TUNE}, Horizon: {HORIZON} days, Trials: {N_TRIALS}")
    
    # --- PERUBAHAN BARU ---
    # Ubah 'direction' menjadi 'maximize'
    study = optuna.create_study(direction='maximize')
    # --- AKHIR PERUBAHAN ---
    
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

    print("\n--- Tuning Complete ---")
    
    if study.best_trial and study.best_value != -float('inf'):
        best_trial = study.best_trial
        print(f"\nBest Trial: {best_trial.number}")
        print(f"  Value (Maximized Sharpe Ratio): {best_trial.value:.4f}")
        
        print("\n  Best Hyperparameters (Model + Strategy):")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
            
        print("\n  Secondary Metrics for Best Trial:")
        print(f"    Max Drawdown: {best_trial.user_attrs.get('max_drawdown')}")

        report_path = Path("artifacts/reports")
        report_path.mkdir(parents=True, exist_ok=True)
        best_params_path = report_path / f"best_sharpe_params_{MODEL_TO_TUNE}_h{HORIZON}.json"
        
        with open(best_params_path, 'w') as f:
            json.dump(best_trial.params, f, indent=4)
            
        print(f"\nBest parameters saved to {best_params_path}")
    else:
        print("\nOptimization finished, but no successful trials were completed.")
        print("Please check the logs for errors in each trial.")