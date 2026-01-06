import pytest
import numpy as np
from aegisx.ta.hurst import hurst_exponent
from aegisx.config import settings

def test_hurst_regimes():
    """Test Hurst relative values on different regimes."""
    np.random.seed(42)
    N = 1000
    
    # 1. Random Walk (Noise)
    rw_ret = np.random.normal(0, 0.01, N)
    rw_prices = 100 * np.exp(np.cumsum(rw_ret))
    h_rw = hurst_exponent(rw_prices, min_lag=2, max_lag=20)
    
    # 2. Momentum (Positive Autocorrelation)
    mom_ret = np.zeros(N)
    eps = np.random.normal(0, 0.01, N)
    for i in range(1, N):
        mom_ret[i] = 0.4 * mom_ret[i-1] + eps[i]
    mom_prices = 100 * np.exp(np.cumsum(mom_ret))
    h_mom = hurst_exponent(mom_prices, min_lag=2, max_lag=20)
    
    # 3. Mean Reversion (Negative Autocorrelation)
    mr_ret = np.zeros(N)
    for i in range(1, N):
        mr_ret[i] = -0.4 * mr_ret[i-1] + eps[i]
    mr_prices = 100 * np.exp(np.cumsum(mr_ret))
    h_mr = hurst_exponent(mr_prices, min_lag=2, max_lag=20)
    
    print(f"\nHurst Values: MR={h_mr:.3f}, RW={h_rw:.3f}, Mom={h_mom:.3f}")
    
    # Expectation: MR < RW < Mom
    assert h_mr < h_rw, f"Mean Rev ({h_mr}) should be < RW ({h_rw})"
    assert h_mom > h_rw, f"Momentum ({h_mom}) should be > RW ({h_rw})"
    
    # Corrected values check
    # RW should be ~ 0.5
    assert 0.4 < h_rw < 0.6
    
    # MR should be < 0.5
    assert h_mr < 0.5
    
    # Mom should be > 0.5 (typically > 0.55)
    assert h_mom > 0.55

def test_kelly_logic():
    """Test Kelly sizing logic independently."""
    # Manual calc of logic used in engine
    # Fraction f = (p(b+1) - 1) / b
    
    # Case 1: High edge
    prob = 0.6
    b = 2.0 # Reward/Risk = 2:1
    
    # f = (0.6(3) - 1) / 2 = (1.8 - 1) / 2 = 0.4
    # Full Kelly = 40%
    # settings.KELLY_FRACTION = 0.5 -> 20%
    # settings.KELLY_MAX_RISK = 0.02 -> Cap at 2%
    
    kf = 0.5
    max_risk = 0.02
    
    kelly_full = (prob * (b + 1) - 1) / b
    risk_pct = min(max_risk, kelly_full * kf)
    
    assert kelly_full == pytest.approx(0.4)
    assert risk_pct == 0.02

    # Case 2: Negative Edge
    prob = 0.3
    b = 2.0
    # Expected Return = 0.3*2 - 0.7 = 0.6 - 0.7 = -0.1
    # Kelly should be negative
    
    kelly_full = (prob * (b + 1) - 1) / b
    # (0.3*3 - 1)/2 = -0.1 / 2 = -0.05
    
    kelly_full = max(0.0, kelly_full)
    assert kelly_full == 0.0
