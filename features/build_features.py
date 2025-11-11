# btc-quant-prob/features/build_features.py

import pandas as pd
import numpy as np
import ta

def add_log_returns(df: pd.DataFrame, lags: list) -> pd.DataFrame:
    """Adds log returns and lagged log returns."""
    df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))
    for lag in lags:
        df[f'log_return_{lag}d'] = df['log_return'].shift(lag)
    return df

def add_volatility_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Adds realized volatility (rolling std of log returns) and ATR."""
    for window in windows:
        # Hitung realized vol, pastikan ada nilai minimum untuk menghindari 0
        df[f'realized_vol_{window}d'] = df['log_return'].rolling(window=window).std() * np.sqrt(365)
    
    df['atr'] = ta.volatility.AverageTrueRange(
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        window=14
    ).average_true_range()
    
    return df

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds RSI, MACD, and Bollinger Bands z-score."""
    df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    macd = ta.trend.MACD(close=df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    
    window_dev = 2
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=window_dev)
    
    mavg = bollinger.bollinger_mavg()
    hband = bollinger.bollinger_hband()
    
    stdev = (hband - mavg) / window_dev
    
    df['bollinger_zscore'] = (df['Close'] - mavg) / (stdev + 1e-9)
    
    return df

def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds On-Balance Volume (OBV)."""
    # Pastikan 'Volume_(BTC)' ada sebelum mencoba menggunakannya
    if 'Volume_(BTC)' in df.columns:
        df['obv'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume_(BTC)']).on_balance_volume()
    return df

def add_statistical_moment_features(df: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Adds rolling skewness and kurtosis of log returns."""
    for window in windows:
        df[f'skew_{window}d'] = df['log_return'].rolling(window=window).skew()
        df[f'kurt_{window}d'] = df['log_return'].rolling(window=window).kurt()
    return df

def add_halving_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds features based on Bitcoin halving cycles.
    """
    halving_dates = [
        pd.to_datetime('2012-11-28'),
        pd.to_datetime('2016-07-09'),
        pd.to_datetime('2020-05-11'),
        pd.to_datetime('2024-04-19')
    ]
    
    df['days_since_last_halving'] = np.nan
    df['days_until_next_halving'] = np.nan
    
    for i in range(len(halving_dates)):
        start_date = halving_dates[i]
        end_date = halving_dates[i+1] if (i+1) < len(halving_dates) else pd.to_datetime('2100-01-01')
        
        mask = (df.index >= start_date) & (df.index < end_date)
        if mask.any():
            df.loc[mask, 'days_since_last_halving'] = (df.loc[mask].index - start_date).days
            df.loc[mask, 'days_until_next_halving'] = (end_date - df.loc[mask].index).days

    pre_halving_mask = df.index < halving_dates[0]
    df.loc[pre_halving_mask, 'days_until_next_halving'] = (halving_dates[0] - df.loc[pre_halving_mask].index).days
    
    # --- PERBAIKAN UNTUK FutureWarning ---
    # Ganti .fillna(0, inplace=True) dengan ini:
    df['days_since_last_halving'] = df['days_since_last_halving'].fillna(0)
    # --- AKHIR PERBAIKAN ---
    
    return df

def add_feature_interactions(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Adds creative feature interactions and ratios to find non-linear patterns.
    """
    vol_windows = config.get('volatility_windows', [30, 90, 365])
    if len(vol_windows) < 2:
        return df 

    short_vol_col = f'realized_vol_{vol_windows[0]}d'
    long_vol_col = f'realized_vol_{vol_windows[-1]}d'

    if short_vol_col in df.columns and long_vol_col in df.columns:
        df['vol_ratio'] = df[short_vol_col] / (df[long_vol_col] + 1e-9)
    
    if 'rsi' in df.columns and short_vol_col in df.columns:
        df['rsi_x_vol'] = df['rsi'] * df[short_vol_col]
        
    if 'bollinger_zscore' in df.columns:
        df['bollinger_zscore_sq'] = df['bollinger_zscore']**2
        
    return df

def build_feature_set(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Main function to build all features.

    Args:
        df (pd.DataFrame): Daily OHLCV data.
        config (dict): Ini adalah config['features'], BUKAN config global.
    """
    df_features = df.copy()
    
    print("Building features (Standard)...")
    df_features = add_log_returns(df_features, config['log_return_lags'])
    df_features = add_volatility_features(df_features, config['volatility_windows'])
    df_features = add_momentum_features(df_features)
    df_features = add_volume_features(df_features)
    df_features = add_statistical_moment_features(df_features, config['rolling_stat_windows'])
    
    df_features['day_of_week'] = df_features.index.dayofweek
    df_features['month_of_year'] = df_features.index.month
    
    print("Building features (Creative Alpha)...")
    df_features = add_halving_features(df_features)
    
    # --- PERBAIKAN UNTUK KeyError: 'features' ---
    # Ganti config['features'] menjadi hanya 'config'
    # karena 'config' sudah merupakan 'config['features']'
    df_features = add_feature_interactions(df_features, config)
    # --- AKHIR PERBAIKAN ---

    df_features.dropna(inplace=True)
    print(f"Feature engineering complete. Shape: {df_features.shape}")
    
    base_cols = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']
    target_cols = [col for col in df_features.columns if 'log_return_' in col]
    target_cols.append('log_return') 
    
    feature_cols = [col for col in df_features.columns if col not in base_cols and col not in target_cols]
    
    df_features[feature_cols] = df_features[feature_cols].replace([np.inf, -np.inf], 0)
    
    return df_features[feature_cols]