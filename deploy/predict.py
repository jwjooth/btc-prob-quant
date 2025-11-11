# btc-quant-prob/deploy/predict.py

import pandas as pd
import numpy as np
import joblib
import json
import argparse
from pathlib import Path
import sys

# --- PENTING: Tambahkan root proyek ke path ---
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ---

from data.ingest import load_and_clean_data
from features.build_features import build_feature_set

def run_prediction(model_name: str, horizon: int, current_price: float):
    """
    Memuat model final dan membuat prediksi probabilistik untuk 180 hari ke depan.
    """
    
    print("--- Menjalankan Prediksi Probabilistik Final ---")
    
    # --- 1. Muat Konfigurasi & Model ---
    config_path = Path('train/config.json')
    model_dir = Path(f"artifacts/models/{model_name}/h{horizon}")
    model_path = model_dir / "model.joblib"
    pipeline_path = model_dir / "pipeline.joblib"
    
    if not model_path.exists() or not pipeline_path.exists():
        print(f"Error: Model artifacts tidak ditemukan di {model_dir}")
        print("Pastikan Anda sudah menjalankan 'python -m train.train ...' dengan config final.")
        return

    print(f"Memuat model dari {model_path}...")
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    
    # --- 2. Muat & Proses Data Terbaru ---
    print("Memuat data historis lengkap...")
    df_daily = load_and_clean_data(config['data']['raw_path'])
    
    print("Membangun fitur terbaru...")
    features_df = build_feature_set(df_daily, config['features'])
    
    latest_features_unscaled = features_df.iloc[[-1]]
    
    print(f"Data fitur terbaru per tanggal: {latest_features_unscaled.index[0].date()}")

    # --- 3. Scaling & Prediksi ---
    
    # --- PERBAIKAN DI SINI ---
    # pipeline.transform() mengembalikan NumPy array.
    # Model kita mengharapkan DataFrame pandas.
    # Kita harus konversi kembali, dengan tetap memakai index dan nama kolom aslinya.
    
    latest_features_scaled_array = pipeline.transform(latest_features_unscaled)
    
    latest_features_scaled = pd.DataFrame(
        latest_features_scaled_array,
        index=latest_features_unscaled.index,
        columns=latest_features_unscaled.columns
    )
    # --- AKHIR PERBAIKAN ---

    # Sekarang 'model.predict' akan menerima DataFrame yang benar
    log_return_preds_df = model.predict(latest_features_scaled)
    
    log_return_preds = log_return_preds_df.iloc[0]
    
    # --- 4. Tampilkan Hasil ---
    quantiles = config['training']['quantiles']
    
    print(f"\n--- Prediksi Probabilistik (Log Return {horizon} hari) ---")
    print(log_return_preds)
    
    print(f"\n--- Prediksi Harga (Basis: ${current_price:,.2f}) ---")
    print(f"{'Kuantil':<12} | {'Log Return':<12} | {'Prediksi Harga':<15}")
    print("-" * 43)
    
    results = {}
    for q, log_r in zip(quantiles, log_return_preds.values):
        future_price = current_price * np.exp(log_r)
        
        log_r_str = f"{log_r:+.2%}"
        price_str = f"${future_price:,.2f}"
        
        results[q] = (log_r_str, price_str)
        print(f"q_{q:<10} | {log_r_str:<12} | {price_str:<15}")

    # --- 5. Tampilkan Interpretasi ---
    print("\n--- Interpretasi untuk Alokasi Portofolio ---")
    print(f"-> Prediksi Median (q_0.5): {results[0.5][1]}")
    print(f"   (Ada 50% probabilitas harga akan di atas/di bawah level ini)")
    print(f"-> Interval Keyakinan 50% (q_0.25 - q_0.75):")
    print(f"   {results[0.25][1]}  ---  {results[0.75][1]}")
    print(f"-> Interval Keyakinan 90% (q_0.05 - q_0.95):")
    print(f"   {results[0.05][1]}  ---  {results[0.95][1]}")
    print("\nGunakan rentang ini untuk menentukan alokasi risiko Anda.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jalankan prediksi probabilistik final.")
    
    parser.add_argument(
        '--price', 
        type=float, 
        required=True, 
        help="Harga Bitcoin SAAT INI (misal: 68000.0)"
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default="lgbm_quantile", 
        help="Nama model yang akan digunakan."
    )
    parser.add_argument(
        '--horizon', 
        type=int, 
        default=180, 
        help="Horizon prediksi (harus sesuai dengan model yang dilatih)."
    )
    
    args = parser.parse_args()
    
    run_prediction(args.model, args.horizon, args.price)