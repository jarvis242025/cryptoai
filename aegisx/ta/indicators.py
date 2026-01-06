import pandas as pd
import numpy as np

def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=length).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Wilder's smoothing: alpha = 1/length
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    
    # Explicitly NaN the first 'length' values to match test expectations
    # First valid RSI is at index 'length'
    rsi_series.iloc[0:length] = np.nan
    return rsi_series

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Moving Average Convergence Divergence"""
    exp1 = ema(series, fast)
    exp2 = ema(series, slow)
    macd_line = exp1 - exp2
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({
        'MACD_12_26_9': macd_line,
        'MACDs_12_26_9': signal_line,
        'MACDh_12_26_9': hist
    })

def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # Wilder's Smoothing (RMA) is standard for ATR, but standard EMA often approximates it well.
    # However, Wilder's smoothing is alpha = 1/length.
    # Pandas ewm com = length - 1 is equivalent to alpha = 1/length.
    return tr.ewm(alpha=1/length, adjust=False).mean()

def bbands(series: pd.Series, length: int = 20, std: int = 2) -> pd.DataFrame:
    """Bollinger Bands"""
    mid = sma(series, length)
    sd = series.rolling(window=length).std()
    upper = mid + (sd * std)
    lower = mid - (sd * std)
    
    # Return structure matching expected access style (iloc)
    return pd.DataFrame({
        'BBL_20_2.0': lower,
        'BBM_20_2.0': mid,
        'BBU_20_2.0': upper
    })
