import pytest
import sys
from unittest.mock import MagicMock
import pandas as pd
import numpy as np

# Only mock network/optional dependencies
sys.modules["ccxt"] = MagicMock()
sys.modules["feedparser"] = MagicMock()

@pytest.fixture
def sample_data():
    """Deterministic sample OHLCV data."""
    dates = pd.date_range(start="2024-01-01", periods=1000, freq="15min")
    np.random.seed(42)
    
    # Generate random walk with drift
    returns = np.random.normal(0, 0.001, 1000)
    price = 100 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        "open": price,
        "high": price * 1.002,
        "low": price * 0.998,
        "close": price * (1 + np.random.normal(0, 0.0005, 1000)),
        "volume": np.random.randint(100, 1000, 1000).astype(float)
    }, index=dates)
    
    return df
