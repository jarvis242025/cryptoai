import pandas as pd
import numpy as np
from aegisx.config import settings

def triple_barrier_label(df: pd.DataFrame) -> pd.Series:
    """
    Generate labels based on Triple Barrier Method.
    1: Profit-take hit before Stop-loss and within Horizon.
    0: Stop-loss hit OR Time limit reached.
    """
    labels = pd.Series(index=df.index, data=0, dtype=int)
    
    # Cost-Aware TP Calculation
    # We want to label '1' only if price hits a target that covers costs and desired profit.
    # Aligning with Engine logic:
    cost_bps = (2 * settings.MAKER_FEE_BPS) + settings.EST_SPREAD_BPS + settings.EST_SLIPPAGE_BPS
    cost_pct = cost_bps / 10000
    
    # Use the same logic as Engine for minimum viable target
    # If TP_PCT is small, we force it to be at least 2x costs.
    # If TP_PCT is large (sniper), it usually exceeds this.
    target_tp_pct = max(settings.TP_PCT, 2.0 * cost_pct)
    
    tp_pct = target_tp_pct
    sl_pct = settings.SL_PCT
    horizon = settings.HORIZON_BARS
    
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    
    n = len(df)
    
    for i in range(n - 1):
        entry_price = closes[i]
        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
        
        outcome = 0 # Default: Timeout/Neutral -> 0
        
        for j in range(1, horizon + 1):
            if i + j >= n:
                break
                
            curr_high = highs[i+j]
            curr_low = lows[i+j]
            
            hit_tp = curr_high >= tp_price
            hit_sl = curr_low <= sl_price
            
            if hit_tp and hit_sl:
                # Ambiguous: both hit in same candle -> 0
                outcome = 0 
                break
            elif hit_tp:
                outcome = 1
                break
            elif hit_sl:
                outcome = 0
                break
        
        labels.iloc[i] = outcome
        
    return labels
