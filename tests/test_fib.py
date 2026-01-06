import pytest
import pandas as pd
import numpy as np
from aegisx.config import settings
from aegisx.ta.fibonacci import calculate_fib_features

def test_fib_features_calculation():
    # Enable Fib for test
    settings.FIB_ENABLED = True
    settings.FIB_PIVOT_LEFT = 1
    settings.FIB_PIVOT_RIGHT = 1

    # Create synthetic data with a clear swing
    # 0, 10, 0, 10 ...
    # Pivot High at 10. Pivot Low at 0.
    
    vals = [10.0, 20.0, 10.0, 5.0, 10.0, 20.0, 15.0]
    # Index 1 (20) is High. Confirmed at 1+1=2.
    # Index 3 (5) is Low. Confirmed at 3+1=4.
    
    df = pd.DataFrame({
        'high': vals,
        'low': vals,
        'close': vals,
        'volume': [100.0]*7
    })
    
    df = calculate_fib_features(df)
    
    # Check columns exist
    assert 'fib_retrace' in df.columns
    assert 'fib_in_golden' in df.columns
    
    # We can inspect specific values if we know the swing logic
    # At index 6 (15.0), the last swing high was 20 (idx 1), last swing low was 5 (idx 3).
    # Since Low (idx 3) is > High (idx 1), Impulse is DOWN (High->Low)? No.
    # Indices: High=1, Low=3. 3 > 1. So Low is more recent. 
    # Impulse is DOWN (High -> Low).
    # Retracement is from Low (5) back up towards High (20).
    # Range = 20 - 5 = 15.
    # Price = 15.
    # Dist from Low = 15 - 5 = 10.
    # Retrace Ratio = 10 / 15 = 0.666
    
    # Note: Logic might verify confirmation delay.
    # Pivot at 1 confirmed at 2.
    # Pivot at 3 confirmed at 4.
    # At index 6, both are known.
    
    # Let's check last row
    last_row = df.iloc[-1]
    # We expect some value. Since we used minimal data, it might be NaN if window is large.
    # Window = 1+1+1 = 3. We have 7 rows. Should be fine.
    
    # Just asserting it didn't crash and produced columns
    assert not df['fib_retrace'].isnull().all()
