import pandas as pd
from typing import List
from aegisx.config import settings
from aegisx.ta.fibonacci import calculate_fib_features
from aegisx.ta.hurst import rolling_hurst
from aegisx.ta.indicators import ema, sma, rsi, macd, atr, bbands

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features for ML model.
    """
    df = df.copy()
    
    # Ensure sufficient data (need ~200 rows for EMA200)
    if len(df) < 200:
        return df
        
    # 1. Trend
    df['ema_20'] = ema(df['close'], length=20)
    df['ema_50'] = ema(df['close'], length=50)
    df['ema_200'] = ema(df['close'], length=200)
    
    # Trend Distances
    df['trend_dist'] = (df['close'] - df['ema_50']) / df['ema_50']
    
    # EMA Slope (pct change of EMA50 over 4 bars)
    df['ema_slope'] = df['ema_50'].pct_change(4)
    
    # 2. Momentum
    df['rsi'] = rsi(df['close'], length=14)
    
    # MACD
    macd_df = macd(df['close'])
    if macd_df is not None:
        # Columns: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        df['macd_hist'] = macd_df['MACDh_12_26_9']
    
    # 3. Volatility
    df['atr'] = atr(df['high'], df['low'], df['close'], length=14)
    df['atr_pct'] = df['atr'] / df['close']
    
    # Bollinger Bands
    bb = bbands(df['close'], length=20, std=2)
    if bb is not None:
        # Use iloc to be safe against version-dependent column names
        # Order: Lower, Mid, Upper
        lower = bb.iloc[:, 0]
        upper = bb.iloc[:, 2]
        df['bb_width'] = (upper - lower) / df['close']
        df['bb_pos'] = (df['close'] - lower) / (upper - lower)
    
    # Rolling Volatility (std dev of returns)
    df['returns'] = df['close'].pct_change()
    df['rolling_vol'] = df['returns'].rolling(20).std()
    
    # Returns windows
    df['returns_1'] = df['close'].pct_change(1)
    df['returns_4'] = df['close'].pct_change(4)
    
    # 4. Volume
    df['vol_ma'] = sma(df['volume'], length=20)
    df['vol_spike'] = df['volume'] / df['vol_ma']
    
    # Vol Z-score
    vol_mean = df['volume'].rolling(50).mean()
    vol_std = df['volume'].rolling(50).std()
    df['vol_z'] = (df['volume'] - vol_mean) / vol_std
    
    # 5. Price Action
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    
    # 6. Fibonacci
    df = calculate_fib_features(df)

    # 7. Hurst Exponent (Regime Filter)
    if settings.HURST_ENABLED:
        df['hurst'] = rolling_hurst(df['close'], window=settings.HURST_WINDOW)
    
    # Clean NaN
    df.dropna(inplace=True)
    
    return df

def get_feature_names() -> List[str]:
    return [
        'trend_dist', 'rsi', 'atr_pct', 'rolling_vol', 
        'vol_spike', 'range_pct',
        'returns_1', 'returns_4', 'ema_slope', 'macd_hist',
        'bb_pos', 'bb_width', 'vol_z',
        'fib_retrace', 'fib_in_golden', 'fib_dist_to_50', 'fib_dist_to_618', 'fib_swing_range_pct',
        'hurst'
    ]
