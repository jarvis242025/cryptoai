import pytest
import pandas as pd
import numpy as np
from aegisx.ta.indicators import sma, ema, rsi, macd, atr, bbands

@pytest.fixture
def sample_data():
    np.random.seed(42)
    close = pd.Series(np.random.randn(100).cumsum() + 100)
    high = close + np.random.rand(100)
    low = close - np.random.rand(100)
    return pd.DataFrame({'close': close, 'high': high, 'low': low})

def test_sma(sample_data):
    s = sma(sample_data['close'], length=10)
    assert len(s) == 100
    assert pd.isna(s[8])  # First 9 should be NaN
    assert not pd.isna(s[9]) # 10th should be valid

def test_ema(sample_data):
    e = ema(sample_data['close'], length=10)
    assert len(e) == 100
    assert not pd.isna(e[0]) # EMA starts from beginning (using close[0] as seed usually, or mean)
    # Pandas ewm with adjust=False starts calculating immediately

def test_rsi(sample_data):
    r = rsi(sample_data['close'], length=14)
    assert len(r) == 100
    assert pd.isna(r[13]) # First 14 are often NaN or unstable
    # With pandas ewm min_periods=length, first length-1 are NaN
    
    # Check bounds
    valid_r = r.dropna()
    assert (valid_r >= 0).all() and (valid_r <= 100).all()

def test_macd(sample_data):
    m = macd(sample_data['close'])
    assert isinstance(m, pd.DataFrame)
    assert 'MACD_12_26_9' in m.columns
    assert 'MACDs_12_26_9' in m.columns
    assert 'MACDh_12_26_9' in m.columns
    assert len(m) == 100

def test_atr(sample_data):
    a = atr(sample_data['high'], sample_data['low'], sample_data['close'], length=14)
    assert len(a) == 100
    # ATR should be positive
    valid_a = a.dropna()
    assert (valid_a > 0).all()

def test_bbands(sample_data):
    bb = bbands(sample_data['close'], length=20, std=2)
    assert isinstance(bb, pd.DataFrame)
    assert len(bb.columns) == 3
    # Check names
    assert 'BBL_20_2.0' in bb.columns
    assert 'BBM_20_2.0' in bb.columns
    assert 'BBU_20_2.0' in bb.columns
    
    # Check logic: Upper > Mid > Lower
    valid_bb = bb.dropna()
    assert (valid_bb['BBU_20_2.0'] >= valid_bb['BBM_20_2.0']).all()
    assert (valid_bb['BBM_20_2.0'] >= valid_bb['BBL_20_2.0']).all()
