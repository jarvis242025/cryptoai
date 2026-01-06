import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from aegisx.ml.train import train_model
from aegisx.config import settings

@patch('aegisx.ml.train.OHLCVFetcher')
def test_train_model_no_crash_on_few_signals(mock_fetcher):
    # Setup mock data
    dates = pd.date_range(start='2023-01-01', periods=300, freq='15min')
    df = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0
    }, index=dates)
    
    mock_fetcher_instance = mock_fetcher.return_value
    mock_fetcher_instance.fetch_recent.return_value = df
    
    # Run
    # Should not crash, just log errors/warnings about no signals
    train_model(days=1, save=False)

@patch('aegisx.ml.train.OHLCVFetcher')
def test_train_model_name_error_regression(mock_fetcher):
    # Provide data that yields some positives to trigger threshold loop
    dates = pd.date_range(start='2023-01-01', periods=1000, freq='15min')
    # Random walk
    df = pd.DataFrame({
        'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0, 'volume': 1000.0
    }, index=dates)
    
    # Inject randomness to get features
    df['close'] = 100 * np.exp(np.random.normal(0, 0.01, 1000).cumsum())
    df['high'] = df['close'] * 1.01
    df['low'] = df['close'] * 0.99
    
    mock_fetcher_instance = mock_fetcher.return_value
    mock_fetcher_instance.fetch_recent.return_value = df
    
    # We want labels to have both classes
    # With random walk, triple barrier might trigger both.
    
    # Ensure MIN_SIGNALS is low so loop runs
    settings.MIN_SIGNALS = 1
    
    train_model(days=1, save=False)
