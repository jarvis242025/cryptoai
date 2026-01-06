import pytest
import pandas as pd
import numpy as np
from aegisx.ml.labeling import triple_barrier_label
from aegisx.config import settings

def test_triple_barrier_tp_hit():
    # Setup: 1% SL, 2% TP
    # Price starts at 100. TP=102, SL=99.
    
    data = {
        'close': [100.0, 100.0, 100.0, 100.0, 100.0],
        'high':  [100.0, 101.0, 103.0, 100.0, 100.0], # Hits 103 at index 2 (relative to 0)
        'low':   [100.0, 99.5, 99.5, 99.5, 99.5]
    }
    df = pd.DataFrame(data)
    
    # Mock settings
    settings.TP_PCT = 0.02
    settings.SL_PCT = 0.01
    settings.HORIZON_BARS = 5
    
    labels = triple_barrier_label(df)
    
    # Index 0: Entry 100.
    # i+1 (1): H=101, L=99.5. No hit.
    # i+2 (2): H=103 (>102). TP Hit!
    # Label should be 1
    assert labels[0] == 1

def test_triple_barrier_sl_hit():
    # Setup: 1% SL, 2% TP
    # Price starts at 100. TP=102, SL=99.
    
    data = {
        'close': [100.0, 100.0, 100.0, 100.0, 100.0],
        'high':  [100.0, 101.0, 101.0, 100.0, 100.0], 
        'low':   [100.0, 99.5, 98.0, 99.5, 99.5] # Hits 98 at index 2
    }
    df = pd.DataFrame(data)
    
    settings.TP_PCT = 0.02
    settings.SL_PCT = 0.01
    settings.HORIZON_BARS = 5
    
    labels = triple_barrier_label(df)
    assert labels[0] == 0 # Loss is 0

def test_triple_barrier_timeout():
    data = {
        'close': [100.0, 100.0, 100.0, 100.0, 100.0],
        'high':  [100.0, 101.0, 101.0, 101.0, 101.0], 
        'low':   [100.0, 99.5, 99.5, 99.5, 99.5]
    }
    df = pd.DataFrame(data)
    
    labels = triple_barrier_label(df)
    assert labels[0] == 0 # Timeout is 0
