# btc-quant-prob/tests/test_data.py

import pytest # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
from pathlib import Path

# Make sure the script can find the 'data' module by adding the project root to the path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.ingest import load_and_clean_data

@pytest.fixture
def mock_csv_file(tmp_path):
    """Creates a mock minute-level data CSV file in a temporary directory for testing."""
    d = tmp_path / "data"
    d.mkdir()
    filepath = d / "mock_btc.csv"
    
    # Create 2 days of minute data
    timestamps = pd.to_datetime(pd.date_range(start='2021-01-01 00:00', periods=2880, freq='T'))
    n = len(timestamps)
    
    df = pd.DataFrame({
        'Timestamp': (timestamps - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'),
        'Open': np.random.uniform(40000, 41000, n),
        'High': np.random.uniform(41000, 42000, n),
        'Low': np.random.uniform(39000, 40000, n),
        'Close': np.random.uniform(40000, 41000, n),
        'Volume_(BTC)': np.random.uniform(1, 10, n),
        'Volume_(Currency)': np.random.uniform(40000, 410000, n),
        'Weighted_Price': np.random.uniform(40000, 41000, n)
    })
    
    df.to_csv(filepath, index=False)
    return str(filepath)

@pytest.fixture
def mock_csv_with_errors(tmp_path):
    """Creates a mock CSV with invalid data (negative volume)."""
    d = tmp_path / "data"
    d.mkdir()
    filepath = d / "mock_btc_error.csv"
    
    timestamps = pd.to_datetime(pd.date_range(start='2021-01-01 00:00', periods=10, freq='T'))
    n = len(timestamps)
    
    df = pd.DataFrame({
        'Timestamp': (timestamps - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s'),
        'Open': [40000] * n, 'High': [41000] * n, 'Low': [39000] * n, 'Close': [40500] * n,
        'Volume_(BTC)': [-5] * n, # Invalid data
        'Volume_(Currency)': [202500] * n, 'Weighted_Price': [40500] * n
    })
    
    df.to_csv(filepath, index=False)
    return str(filepath)

def test_load_and_clean_data_successful(mock_csv_file):
    """Tests successful loading, cleaning, and resampling of data."""
    df_daily = load_and_clean_data(mock_csv_file)
    
    assert isinstance(df_daily, pd.DataFrame)
    assert not df_daily.empty
    assert isinstance(df_daily.index, pd.DatetimeIndex)
    
    # Check if data is resampled to daily frequency. Should have 2 days.
    assert len(df_daily) == 2
    assert all(df_daily.index.to_series().diff().dt.days.dropna() == 1)
    
    # Check for expected columns
    expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume_(BTC)', 'Volume_(Currency)']
    assert all(col in df_daily.columns for col in expected_cols)
    
    # Check for no missing values after forward-fill
    assert not df_daily.isnull().values.any()

def test_load_data_raises_error_on_negative_volume(mock_csv_with_errors):
    """Tests that the function raises a ValueError for invalid data."""
    with pytest.raises(ValueError, match="Data contains negative volume"):
        load_and_clean_data(mock_csv_with_errors)

def test_load_data_handles_missing_file_by_mocking():
    """
    Tests that a mock data file is generated if the source file is not found.
    This ensures the system can run end-to-end without needing the real dataset.
    """
    # Provide a path to a non-existent file in a non-existent directory
    non_existent_path = "temp_test_data_dir/non_existent_file.csv"
    
    p = Path(non_existent_path)
    if p.exists(): p.unlink()
    if p.parent.exists(): p.parent.rmdir()

    # The function should catch the FileNotFoundError, print a warning, and generate a mock file
    df_daily = load_and_clean_data(non_existent_path)
    
    assert isinstance(df_daily, pd.DataFrame)
    assert not df_daily.empty
    assert Path(non_existent_path).exists() # Check that the mock file was created
    
    # Clean up the created mock file and directory
    p.unlink()
    p.parent.rmdir()