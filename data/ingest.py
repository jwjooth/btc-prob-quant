import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
import os
import json

# --- KONFIGURASI PENGUNDUHAN OTOMATIS ---
TICKER = "BTC-USD"
START_DATE_PROJECT = "2012-01-01" 
YF_START_DATE = "2014-09-17" 
# ----------------------------------------

def load_and_clean_data(raw_path: str) -> pd.DataFrame:
    """
    Memuat data historis BTC/USD dari file lokal, secara otomatis mengunduh 
    data terbaru dari Yahoo Finance, menggabungkan, dan menyimpan data yang diperbarui.
    """
    raw_path_obj = Path(raw_path)
    df_old = pd.DataFrame()
    df_new = pd.DataFrame() # Inisialisasi df_new
    start_download_date = pd.to_datetime(START_DATE_PROJECT)
    
    # 2. Coba memuat data historis yang sudah ada
    if raw_path_obj.exists():
        try:
            print(f"Loading existing data from {raw_path}...")
            df_old = pd.read_csv(raw_path_obj)
            
            # --- PERBAIKAN KOLOM TANGGAL (Toleran) ---
            date_col = 'Timestamp'
            try:
                # Coba parse Unix
                df_old[date_col] = pd.to_datetime(df_old[date_col], unit='s')
            except (KeyError, ValueError, TypeError):
                 # Jika gagal, coba parse String
                 date_col = 'Date' if 'Date' in df_old.columns else df_old.index.name
                 df_old.index = pd.to_datetime(df_old[date_col] if date_col and date_col in df_old.columns else df_old.index)
                 
            if date_col in df_old.columns:
                 df_old.set_index(date_col, inplace=True)
            # --- AKHIR PERBAIKAN TANGGAL ---
            
            df_old = df_old.sort_index()
            df_old = df_old[~df_old.index.duplicated(keep='last')]
            
            if not df_old.empty:
                # Pastikan index bertipe DatetimeIndex sebelum normalize
                if not isinstance(df_old.index, pd.DatetimeIndex):
                     df_old.index = pd.to_datetime(df_old.index, errors='coerce')
                     # Hapus baris dengan index yang gagal parse (termasuk 1970-01-01 yang rusak)
                     df_old = df_old[df_old.index.notna()]
                     
                if df_old.empty:
                    raise ValueError("Data lama mengandung terlalu banyak tanggal rusak.")

                last_date = df_old.index.max().normalize()
                start_download_date = last_date + timedelta(days=1)
                
                if 'Volume' in df_old.columns and 'Volume_(BTC)' not in df_old.columns:
                     df_old.rename(columns={'Volume': 'Volume_(BTC)'}, inplace=True)
                     
                print(f"Existing data ends on: {last_date.date()}. Downloading updates from {start_download_date.date()}...")
            
        except Exception as e:
            # PENTING: Jika gagal membaca atau jika data lama tidak valid (masuk ke raise ValueError di atas)
            print(f"Warning: Failed to read or parse old data file. Error: {e}. Starting fresh download from {YF_START_DATE}.")
            df_old = pd.DataFrame()
            # SETELAH GAGAL, KITA HARUS MENGUNDUH DARI TANGGAL MULAI YANG VALID (2014-09-17)
            start_download_date = pd.to_datetime(YF_START_DATE) # <-- INI PERBAIKAN KRITISNYA
    else:
        print(f"Raw data file not found at {raw_path}. Starting fresh download from {YF_START_DATE}...")
        start_download_date = pd.to_datetime(YF_START_DATE) # <-- JUGA HARUS MENGGUNAKAN YF_START_DATE
    
    # 3. Unduh data terbaru menggunakan yfinance
    end_download_date = datetime.now().date()
    final_download_start_date = max(pd.to_datetime(YF_START_DATE).date(), start_download_date.date())
    
    if final_download_start_date <= end_download_date:
        try:
            df_new = yf.download(
                TICKER, 
                start=final_download_start_date.strftime('%Y-%m-%d'), 
                end=(end_download_date + timedelta(days=1)).strftime('%Y-%m-%d'),
                progress=False
            )
            if not df_new.empty:
                 print(f"Downloaded {len(df_new)} new rows up to {df_new.index.max().date()}.")
            
        except Exception as e:
            print(f"FATAL ERROR: Failed to download data from yfinance. Error: {e}")
            df_new = pd.DataFrame() # Pastikan df_new adalah DF kosong jika gagal

    # 4. Gabungkan dan Bersihkan
    df_combined = pd.DataFrame() # Inisialisasi df_combined di awal

    if not df_new.empty:
        df_new.rename(columns={'Adj Close': 'Close', 'Volume': 'Volume_(BTC)'}, inplace=True)
        df_new = df_new[['Open', 'High', 'Low', 'Close', 'Volume_(BTC)']]
        
        if not df_old.empty:
            df_combined = pd.concat([df_old, df_new])
            df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        else:
            df_combined = df_new
            
    elif not df_old.empty:
        df_combined = df_old
        
    else:
        # Jika tidak ada data lama maupun baru, kita TIDAK raise error, 
        # tapi mengembalikan DataFrame kosong agar program calling tidak crash.
        print("Warning: Could not load data, and new download failed. Returning empty DataFrame.")
        return pd.DataFrame() # Return DataFrame kosong (SOLUSI UNTUK NONETYPE)
        

    # 5. Finalisasi, Resampling Harian, dan Penyimpanan
    # Lakukan semua langkah ini HANYA JIKA df_combined TIDAK KOSONG
    if not df_combined.empty:
        df_combined = df_combined.sort_index().asfreq('D')

        # Isi hari-hari kosong (OHLC) menggunakan .loc dan ffill()
        OHLC_COLS = ['Open', 'High', 'Low', 'Close']
        df_combined.loc[:, OHLC_COLS] = df_combined[OHLC_COLS].ffill()

        # Perbaikan Final Volume: Cast ke float/numeric sebelum fillna untuk menghindari FutureWarning
        df_combined.loc[:, 'Volume_(BTC)'] = pd.to_numeric(df_combined['Volume_(BTC)'], errors='coerce')
        df_combined.loc[:, 'Volume_(BTC)'] = df_combined['Volume_(BTC)'].fillna(0).astype(np.int64, errors='ignore')

        # Simpan data yang sudah diperbarui (Menimpa file lama)
        df_combined.to_csv(raw_path_obj)
        print(f"Data successfully updated and saved to {raw_path_obj}.")

        # Laporan Status
        print("Data ingestion and daily resampling complete.")
        print(f"Data shape: {df_combined.shape}")
        print(f"Date range: {df_combined.index.min().date()} to {df_combined.index.max().date()}")

        # Cek Keamanan
        if len(df_combined) < 1000:
             # Ini mungkin terjadi jika data yfinance tiba-tiba hilang.
             print("Warning: Data size is unexpectedly small. Check yfinance connection.")
        
        return df_combined

    # Ini adalah FALLBACK return jika df_combined kosong setelah semua proses
    return pd.DataFrame()