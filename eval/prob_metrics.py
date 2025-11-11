# btc-quant-prob/eval/prob_metrics.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def crps(y_true, quantile_preds, quantiles):
    """
    Calculates the Continuous Ranked Probability Score (CRPS)
    using the Pinball Loss function, averaged over all quantiles.
    """
    y_true_tiled = np.tile(y_true.reshape(-1, 1), (1, len(quantiles)))
    quantiles_tiled = np.tile(quantiles, (len(y_true), 1))
    
    loss_values = np.maximum(
        quantiles_tiled * (y_true_tiled - quantile_preds),
        (1 - quantiles_tiled) * (quantile_preds - y_true_tiled)
    )
    
    score = np.mean(loss_values)
    return score

def picp(y_true, lower_bound, upper_bound):
    """
    Calculates the Prediction Interval Coverage Probability (PICP).
    """
    return np.mean((y_true >= lower_bound) & (y_true <= upper_bound))

def winkler_score(y_true, lower_bound, upper_bound, alpha):
    """
    Calculates the Winkler Score. Penalizes for wide intervals and misses.
    """
    score = np.where(
        y_true < lower_bound,
        (upper_bound - lower_bound) + (2 / alpha) * (lower_bound - y_true),
        np.where(
            y_true > upper_bound,
            (upper_bound - lower_bound) + (2 / alpha) * (y_true - upper_bound),
            (upper_bound - lower_bound)
        )
    )
    return np.mean(score)
    
def plot_reliability_diagram(y_true, quantile_preds, quantiles):
    """Plots a reliability diagram to assess calibration."""
    plt.figure(figsize=(8, 8))
    observed_freq = []
    for i, q in enumerate(quantiles):
        observed_freq.append(np.mean(y_true <= quantile_preds[:, i]))
    
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(quantiles, observed_freq, 'o-', label='Model Calibration')
    plt.xlabel('Forecasted Quantile (Probability)')
    plt.ylabel('Observed Frequency')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.show()

def evaluate_probabilistic_forecasts(predictions: pd.DataFrame, quantiles: list):
    """
    Runs all probabilistic evaluation metrics.
    """
    y_true = predictions.iloc[:, 0].values
    
    # --- PERBAIKAN DI SINI ---
    # Kita tidak bisa lagi pakai iloc. Kita harus pilih kolom 'q_' secara eksplisit.
    quantile_cols = [col for col in predictions.columns if col.startswith('q_')]
    if len(quantile_cols) != len(quantiles):
        raise ValueError(f"Jumlah kolom kuantil ({len(quantile_cols)}) tidak cocok "
                         f"dengan jumlah kuantil di config ({len(quantiles)}).")
    
    quantile_preds = predictions[quantile_cols].values
    # --- AKHIR PERBAIKAN ---

    results = {}
    
    results['CRPS'] = crps(y_true, quantile_preds, quantiles)
    
    # --- PERBAIKAN DI SINI JUGA ---
    # Kita tidak bisa lagi pakai iloc[-1] karena itu adalah 'Close'.
    # Kita harus cari nama kolom kuantil bawah dan atas secara dinamis.
    alpha = (1 - quantiles[-1]) + quantiles[0]
    nominal_coverage = 1 - alpha
    lower_bound_col = f'q_{quantiles[0]}'  # Hasilnya 'q_0.05'
    upper_bound_col = f'q_{quantiles[-1]}' # Hasilnya 'q_0.95'
    
    lower_bound = predictions[lower_bound_col].values
    upper_bound = predictions[upper_bound_col].values
    # --- AKHIR PERBAIKAN ---
    
    results['PICP'] = picp(y_true, lower_bound, upper_bound)
    results['Winkler_Score'] = winkler_score(y_true, lower_bound, upper_bound, alpha)
    
    print(f"--- Probabilistic Evaluation (Nominal Coverage: {nominal_coverage:.0%}) ---")
    print(f"CRPS: {results['CRPS']:.4f}")
    print(f"PICP: {results['PICP']:.3f}")
    print(f"Winkler Score: {results['Winkler_Score']:.4f}")
    
    plot_reliability_diagram(y_true, quantile_preds, np.array(quantiles))
    
    return results