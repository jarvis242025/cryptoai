import pandas as pd
import numpy as np
import logging
from aegisx.config import settings

logger = logging.getLogger(__name__)

def calculate_fib_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append Fibonacci retracement features to the DataFrame without lookahead bias.
    
    Logic:
    - A pivot at T is confirmed at T + R (Right shoulder bars).
    - We identify pivots using a rolling window of size L + R + 1.
    - We track the 'last confirmed swing high' and 'last confirmed swing low' available at time T.
    - We calculate the retracement ratio based on the most recent impulse direction.
    """
    if not settings.FIB_ENABLED:
        # Return safe defaults if disabled but features are expected?
        # Ideally we should fill with 0 or NaN.
        # But if model expects them, we must provide them.
        # If enabled is false, model might not use them if not trained with them.
        # But to be safe for feature list consistency:
        for col in ['fib_retrace', 'fib_in_golden', 'fib_dist_to_50', 'fib_dist_to_618', 'fib_swing_range_pct']:
            df[col] = 0.0
        return df

    l = settings.FIB_PIVOT_LEFT
    r = settings.FIB_PIVOT_RIGHT
    window = l + r + 1
    
    # Identify Pivots (Rolling Max/Min)
    # roll_max[i] is max of [i-(window-1), i]
    # We check if high[i-r] == roll_max[i]
    
    roll_max = df['high'].rolling(window).max()
    roll_min = df['low'].rolling(window).min()
    
    # Candidate values at i-r
    candidate_high = df['high'].shift(r)
    candidate_low = df['low'].shift(r)
    
    # Pivot confirmation mask (at time i)
    # Note: We must handle NaNs at the start.
    is_pivot_high = (candidate_high == roll_max)
    is_pivot_low = (candidate_low == roll_min)
    
    # Create Series to hold the confirmed pivot values and their occurrence 'time' (integer index)
    # We use integer index to compare recency easily.
    # We can use df.index if it's datetime, but ints are safer for comparison.
    idx_series = pd.Series(np.arange(len(df)), index=df.index)
    
    # --- Swing Highs ---
    # Where confirmed, value is candidate_high. Else NaN. Then FFill.
    confirmed_highs = candidate_high.where(is_pivot_high)
    last_swing_high = confirmed_highs.ffill()
    
    # Time of swing high
    confirmed_high_idxs = idx_series.shift(r).where(is_pivot_high)
    last_swing_high_idx = confirmed_high_idxs.ffill()
    
    # --- Swing Lows ---
    confirmed_lows = candidate_low.where(is_pivot_low)
    last_swing_low = confirmed_lows.ffill()
    
    confirmed_low_idxs = idx_series.shift(r).where(is_pivot_low)
    last_swing_low_idx = confirmed_low_idxs.ffill()
    
    # --- Trend Direction ---
    # If high_idx > low_idx: Impulse UP (Low -> High). Retracing Down.
    # If low_idx > high_idx: Impulse DOWN (High -> Low). Retracing Up.
    
    impulse_up = last_swing_high_idx > last_swing_low_idx
    
    # Calculate Retrace
    # If impulse UP: Range = High - Low. Retrace = (High - Current) / Range.
    # If impulse DOWN: Range = High - Low. Retrace = (Current - Low) / Range.
    # Current price: usually Close.
    
    swing_range = last_swing_high - last_swing_low
    # Avoid div/0
    swing_range = swing_range.replace(0, np.nan)
    
    # Initialize features
    fib_retrace = pd.Series(0.0, index=df.index)
    
    # Vectorized calculation
    # Mask where range is valid
    valid_mask = swing_range.notna()
    
    # Impulse UP case
    up_mask = valid_mask & impulse_up
    fib_retrace[up_mask] = (last_swing_high[up_mask] - df['close'][up_mask]) / swing_range[up_mask]
    
    # Impulse DOWN case
    down_mask = valid_mask & (~impulse_up)
    fib_retrace[down_mask] = (df['close'][down_mask] - last_swing_low[down_mask]) / swing_range[down_mask]
    
    # Features
    df['fib_retrace'] = fib_retrace.fillna(0.0) # 0 if no swing found yet
    
    # In Golden Zone (0.5 - 0.618)
    in_golden = (fib_retrace >= settings.FIB_GOLDEN_LOW) & (fib_retrace <= settings.FIB_GOLDEN_HIGH)
    df['fib_in_golden'] = in_golden.astype(int)
    
    # Distances
    # Pct difference from price level of 0.5/0.618
    # Level UP: Low + 0.5 * Range (No, retrace is from High down).
    # If Up (L->H): 0.5 Retrace Level = High - 0.5 * Range.
    # If Down (H->L): 0.5 Retrace Level = Low + 0.5 * Range.
    # Actually, 0.5 is same from both sides.
    # 0.618 Level: 
    # UP: High - 0.618 * Range.
    # DOWN: Low + 0.618 * Range.
    
    # Let's compute levels explicitly? Or just distances in 'retrace' space?
    # Distance in retrace space is linear to price pct if range is normalized.
    # Request: "fib_dist_to_50, fib_dist_to_618 (as pct of price)"
    
    # Level calc
    fib_50_level = pd.Series(np.nan, index=df.index)
    fib_618_level = pd.Series(np.nan, index=df.index)
    
    fib_50_level[up_mask] = last_swing_high[up_mask] - 0.5 * swing_range[up_mask]
    fib_618_level[up_mask] = last_swing_high[up_mask] - 0.618 * swing_range[up_mask]
    
    fib_50_level[down_mask] = last_swing_low[down_mask] + 0.5 * swing_range[down_mask]
    fib_618_level[down_mask] = last_swing_low[down_mask] + 0.618 * swing_range[down_mask]
    
    df['fib_dist_to_50'] = (df['close'] - fib_50_level) / df['close']
    df['fib_dist_to_618'] = (df['close'] - fib_618_level) / df['close']
    
    df['fib_dist_to_50'] = df['fib_dist_to_50'].fillna(0.0)
    df['fib_dist_to_618'] = df['fib_dist_to_618'].fillna(0.0)
    
    df['fib_swing_range_pct'] = (swing_range / df['close']).fillna(0.0)
    
    # Expose raw levels for Engine (Dynamic Brackets)
    df['fib_swing_high'] = last_swing_high
    df['fib_swing_low'] = last_swing_low
    df['fib_impulse_up'] = impulse_up.astype(int)
    
    return df

def get_fib_levels(swing_low: float, swing_high: float, direction: str) -> dict:
    """
    Compute Fib levels for dynamic brackets.
    Direction 'UP' means impulse was UP (Low -> High), we are looking for entry on retrace DOWN?
    Wait. Dynamic brackets are for determining TP/SL *after* entry.
    If we enter LONG:
      We expect price to go UP.
      If we entered on a retrace of an UP move (Trend Following):
        We target Extensions above the High.
        TPs: High, High + 0.272(Range), etc.
        SL: Below Low? Or below retrace?
      If we entered on a retrace of a DOWN move (Counter Trend):
        Dangerous. But typically we trade with trend.
    
    Assumption: We trade WITH the impulse.
    If Impulse UP (Low->High), we enter near retrace. Target > High.
    """
    rng = swing_high - swing_low
    levels = {}
    
    # Extensions
    # 1.0 = High (if UP)
    # 1.272 = High + 0.272 * Range
    # 1.618 = High + 0.618 * Range
    
    # Only supporting LONG entries for now in Engine logic?
    # If direction == 'UP' (Impulse was UP), we target extensions above High.
    
    if direction == 'UP':
        for ext in settings.FIB_TP_EXTENSIONS:
            if ext == 1.0:
                price = swing_high
            else:
                price = swing_high + (ext - 1.0) * rng
            levels[f'ext_{ext}'] = price
            
        # SL levels
        # Conservative: Below 0.618 retrace (High - 0.618*R)
        # Deep: Below Low
        levels['sl_618'] = swing_high - 0.65 * rng # Buffer
        levels['sl_low'] = swing_low - 0.01 * rng # Buffer
        
    # If SHORT supported later...
    
    return levels
