# btc-quant-prob/backtest/simulate.py

import pandas as pd
import numpy as np

def simulate_trading(predictions: pd.DataFrame, transaction_cost: float, k_value: float = 0.813779473743316): # <-- UBAH DI SINI
    """
    Simulates a CONTRARIAN strategy with:
    1. Volatility Sizing (Risk V1)
    2. Non-Linear (tanh) Position Sizing (Risk V2 / Alpha Scaling)
    """
    
    actual_return_col = predictions.columns[0]

    # --- 1. RISK MANAGEMENT: VOLATILITY SIZING (V3) ---
    # "Ambil posisi lebih kecil saat model tidak yakin (interval lebar)"
    prediction_uncertainty = predictions['q_0.95'] - predictions['q_0.05']
    prediction_uncertainty[prediction_uncertainty <= 0] = np.nan
    prediction_uncertainty.ffill(inplace=True)
    prediction_uncertainty.fillna(1e-9, inplace=True)

    target_uncertainty = prediction_uncertainty.median()
    # 'vol_position_size' adalah skala risiko kita (misal 0.25x s/d 1.0x)
    vol_position_size = target_uncertainty / prediction_uncertainty
    vol_position_size = vol_position_size.clip(lower=0.25, upper=1.0)
    
    # --- 2. ALPHA SCALING: NON-LINEAR (TANH) SIZING ---
    # "Ambil posisi lebih besar saat model LEBIH YAKIN (dan salah)"
    
    # Buat skor keyakinan untuk sinyal kontrarian kita
    # Skor positif = sinyal long (karena q_0.75 sangat negatif)
    # Skor negatif = sinyal short (karena q_0.25 sangat positif)
    long_conviction  = (-0.1) - predictions['q_0.75']
    short_conviction = predictions['q_0.25'] - (0.1)
    
    # Gabungkan menjadi satu skor.
    # Jika q_0.75 < -0.1, skor > 0. Jika q_0.25 > 0.1, skor < 0.
    final_conviction = long_conviction - short_conviction
    final_conviction.fillna(0, inplace=True)

    # Standarisasi skor keyakinan (Z-score) agar skalanya konsisten
    conviction_std = final_conviction.std()
    if conviction_std == 0: conviction_std = 1 # Hindari pembagian dengan nol
    
    scaled_conviction = final_conviction / conviction_std

    # 'k' adalah parameter "agresivitas". k kecil = sinyal lambat. k besar = sinyal cepat.
    k = k_value
    
    # Gunakan np.tanh (Sigmoid -1 s/d +1) untuk mengubah skor keyakinan
    # menjadi ukuran posisi alfa (alpha_signal) yang mulus.
    # Ini adalah INTI dari portfolio sizing kita.
    alpha_signal = np.tanh(scaled_conviction * k)
    
    # --- 3. GABUNGKAN SEMUANYA ---
    # Posisi akhir = (Sinyal Alpha Skala -1 s/d +1) * (Ukuran Risiko 0.25 s/d 1.0)
    final_positions = alpha_signal * vol_position_size
    
    final_positions = final_positions.ffill().fillna(0)

    # --- Kalkulasi Return (Loop harian untuk akurasi biaya trading) ---
    returns = pd.Series(index=predictions.index, dtype=float)
    daily_return_proxy = predictions[actual_return_col].diff().fillna(0)
    
    for i in range(len(predictions)):
        # Ambil posisi dari 'kemarin'
        prev_position = final_positions.iloc[i-1] if i > 0 else 0
        current_position = final_positions.iloc[i]

        # Biaya trading HANYA jika posisi berubah
        trade = 0
        if i > 0 and current_position != prev_position:
            trade = np.abs(current_position - prev_position)
            
        daily_ret = (prev_position * daily_return_proxy.iloc[i]) - (trade * transaction_cost)
        returns.iloc[i] = daily_ret
        
    strategy_returns = returns.fillna(0)
    return strategy_returns

# --- calculate_performance_metrics (Dengan perbaikan log-return) ---
def calculate_performance_metrics(returns: pd.Series):
    """Calculates key performance metrics."""
    if returns.empty or returns.std() == 0:
        return {
            'Total Return': '0.00%', 'Annualized Return': '0.00%',
            'Annualized Volatility': '0.00%', 'Sharpe Ratio': '0.00',
            'Max Drawdown': '0.00%'
        }
        
    log_returns = np.log(1 + returns.replace([np.inf, -np.inf], 0))
    log_returns.fillna(0, inplace=True)
    
    total_return = np.exp(log_returns.sum()) - 1
    annualized_return = np.exp(log_returns.mean() * 365) - 1
    annualized_volatility = log_returns.std() * np.sqrt(365)
    
    sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
    
    cumulative_returns = np.exp(log_returns.cumsum())
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    metrics = {
        'Total Return': f"{total_return:.2%}",
        'Annualized Return': f"{annualized_return:.2%}",
        'Annualized Volatility': f"{annualized_volatility:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}"
    }
    return metrics

if __name__ == '__main__':
    try:
        preds = pd.read_parquet('artifacts/backtest/lgbm_quantile_h180_predictions.parquet')
        strategy_returns = simulate_trading(preds, transaction_cost=0.001)
        performance = calculate_performance_metrics(strategy_returns)
        print("--- Trading Simulation Performance (V4: Non-Linear Sizing) ---")
        for key, value in performance.items():
            print(f"{key}: {value}")
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}")