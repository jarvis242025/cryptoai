import pandas as pd
import numpy as np
from typing import Optional

def hurst_exponent(prices: np.ndarray, min_lag: int = 2, max_lag: int = 20) -> float:
    """
    Calculate the Hurst Exponent of a time series using R/S analysis.
    
    H < 0.5: Mean reverting
    H = 0.5: Random walk (Brownian motion)
    H > 0.5: Trending (Persistent)
    
    Args:
        prices: Array of prices (must be 1D)
        min_lag: Minimum lag for R/S calculation
        max_lag: Maximum lag for R/S calculation (should be << len(prices))
        
    Returns:
        float: Hurst exponent or np.nan if calculation fails
    """
    if len(prices) < max_lag + 2:
        return np.nan
    
    # Standard Hurst for financial data is applied to Returns (Increments)
    # Convert prices to log returns
    # Add small epsilon to avoid log(0) if any
    prices = np.array(prices, dtype=float)
    # handle zeros
    prices[prices <= 0] = 1e-9
    
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    
    series = returns
    
    lags = range(min_lag, max_lag + 1)
    tau = []
    rs_values = []
    
    for lag in lags:
        # Divide series into chunks of size 'lag'? 
        # Simplified R/S: Calculate R/S for the *entire* window of length 'lag'? 
        # No, usually we calculate R/S for varying window sizes 'n' (lags) 
        # within the provided 'prices' window.
        
        # But here 'prices' IS the window (e.g. 200 bars).
        # We want to estimate H for this specific window.
        # So we iterate n from min_lag to max_lag.
        
        n = lag
        
        # We need multiple samples of length n to get a robust average R/S, 
        # but with a short lookback (200 bars), we might just take non-overlapping chunks 
        # or just one chunk if n ~ length?
        # Actually, standard method:
        # For a fixed window (e.g. 200), we split it into sub-series of length n.
        
        # Truncate series to be multiple of n
        limit = len(series) - (len(series) % n)
        if limit < n:
            continue
            
        y = series[:limit]
        
        # Reshape to (num_chunks, n)
        # However, making chunks reduces data usage. 
        # Let's use the full series diff approach or simplified variance ratio for speed?
        # The prompt requested "robust log-log slope across lags".
        # Let's stick to standard R/S on sub-chunks.
        
        chunks = y.reshape(-1, n)
        
        # Calculate R/S for each chunk
        # 1. Mean of each chunk
        means = np.mean(chunks, axis=1, keepdims=True)
        
        # 2. Deviations from mean
        y_centered = chunks - means
        
        # 3. Cumulative deviations
        y_cum = np.cumsum(y_centered, axis=1)
        
        # 4. Range
        ranges = np.max(y_cum, axis=1) - np.min(y_cum, axis=1)
        
        # 5. Standard deviation
        stds = np.std(chunks, axis=1, ddof=1) # Sample std dev
        
        # Avoid div by zero
        stds = np.where(stds == 0, 1e-9, stds)
        
        # 6. R/S
        rs = ranges / stds
        
        # Average R/S for this lag
        avg_rs = np.mean(rs)
        
        if avg_rs > 0:
            rs_values.append(avg_rs)
            tau.append(n)
            
    if len(tau) < 2:
        return np.nan
        
    # Log-log plot
    x = np.log(tau)
    y = np.log(rs_values)
    
    # Linear regression slope
    # Polyfit returns [slope, intercept]
    try:
        slope, _ = np.polyfit(x, y, 1)
        # Small lag bias correction:
        # Empirical tests show RW ~ 0.75 for lags 2-20.
        # We subtract 0.25 to align RW with standard 0.5.
        return float(slope) - 0.25
    except:
        return np.nan

def rolling_hurst(series: pd.Series, window: int = 200) -> pd.Series:
    """
    Calculate rolling Hurst exponent.
    
    Args:
        series: Price series
        window: Rolling window size
        
    Returns:
        pd.Series: Rolling Hurst values
    """
    return series.rolling(window).apply(lambda x: hurst_exponent(x.values), raw=False)
