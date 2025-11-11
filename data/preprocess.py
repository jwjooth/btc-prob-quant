# GANTI SELURUH ISI FILE DENGAN KODE INI

import pandas as pd
import numpy as np  # <-- TAMBAHAN BARIS INI
from sklearn.preprocessing import StandardScaler

def create_target(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """
    Creates the target variable: future log return over a given horizon.

    Args:
        df (pd.DataFrame): DataFrame with 'Close' prices.
        horizon_days (int): The forecast horizon in days.

    Returns:
        pd.Series: The future log return series.
    """
    target_name = f'log_return_{horizon_days}d'
    # --- PERBAIKAN DI SINI: ganti pd.np.log menjadi np.log ---
    df[target_name] = np.log(df['Close'].shift(-horizon_days) / df['Close'])
    return df[target_name]

def scale_features(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    Scales features using StandardScaler.

    Args:
        X_train (pd.DataFrame): Training features.
        X_test (pd.DataFrame): Testing features.

    Returns:
        tuple: Scaled X_train, scaled X_test, and the fitted scaler object.
    """
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
    return X_train_scaled, X_test_scaled, scaler

def split_data(features: pd.DataFrame, target: pd.Series, split_date: str):
    """
    Splits data into training and testing sets based on a date.

    Args:
        features (pd.DataFrame): The feature matrix.
        target (pd.Series): The target series.
        split_date (str): The date to split on (e.g., '2020-01-01').

    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    train_mask = features.index < pd.to_datetime(split_date)
    test_mask = features.index >= pd.to_datetime(split_date)

    X_train, X_test = features[train_mask], features[test_mask]
    y_train, y_test = target[train_mask], target[test_mask]

    # Drop NaNs that might result from target creation
    y_train.dropna(inplace=True)
    y_test.dropna(inplace=True)
    X_train = X_train.loc[y_train.index]
    X_test = X_test.loc[y_test.index]

    return X_train, X_test, y_train, y_test