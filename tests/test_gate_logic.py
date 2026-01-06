import pytest
from unittest.mock import MagicMock, patch
from aegisx.engine.engine import TradingEngine, EngineState
from aegisx.config import settings
from aegisx.broker.sim import SimBroker

# Since pandas is mocked, we use dicts or mocks for rows.
# The code expects row.get() and row['key']. A dict works.

@pytest.fixture
def mock_engine():
    broker = SimBroker()
    with patch("aegisx.engine.engine.Predictor._load_latest_model", return_value=(MagicMock(), {})):
        engine = TradingEngine(broker, mode="sim", force=True)
    engine.predictor = MagicMock()
    engine.predictor.predict_probability.return_value = 0.5
    engine.anomaly = MagicMock()
    engine.anomaly.is_anomaly.return_value = False
    
    # Disable Hurst/News/Fib for base tests to isolate logic
    return engine

def test_funnel_logic(mock_engine):
    """Test that gates only trigger if candidate (Trend + Prob) is true."""
    # Mock row as dict
    row = {
        'close': 100, 
        'ema_50': 90, # Trend OK
        'fib_in_golden': 0,
        'hurst': 0.6,
        'returns_1': 0.01,
        'vol_z': 0.0
    }
    # Allow .name access (timestamp)
    # We can wrap dict in a class or just assign attr to a mock
    # But for now, row is passed to tick(). tick() accesses row.name in loggers?
    # No, tick() accesses row.name inside _enter_trade logger?
    # Actually tick() calls `row.name` for logging `Signal at {row.name}`.
    # So we need an object.
    
    class MockRow(dict):
        @property
        def name(self): return "2025-01-01"
    
    row_obj = MockRow(row)
    
    # 1. Case: Prob Low -> Not Candidate -> No Blocks Counted
    mock_engine.threshold = 0.6 # High threshold
    mock_engine.predictor.predict_probability.return_value = 0.4
    
    settings.HURST_ENABLED = True
    settings.HURST_MIN = 0.5 # Hurst is 0.6, so OK
    
    mock_engine.tick(row_obj)
    
    d = mock_engine.diagnostics
    assert d['trend_ok_count'] == 1
    assert d['prob_pass_count'] == 0
    assert d['candidate_count'] == 0
    assert d['blocked_hurst_count'] == 0 

    # 2. Case: Prob High -> Candidate -> Hurst Blocked
    mock_engine.predictor.predict_probability.return_value = 0.7
    row_obj['hurst'] = 0.4 # Below min 0.52
    settings.HURST_MIN = 0.52
    
    mock_engine.tick(row_obj)
    
    d = mock_engine.diagnostics
    assert d['candidate_count'] == 1
    assert d['blocked_hurst_count'] == 1
    assert d['entries_count'] == 0

def test_threshold_precedence():
    """Test Override > Config > Model precedence."""
    settings.ENTRY_THRESHOLD = 0.6
    settings.ENTRY_THRESHOLD_OVERRIDE = None
    settings.AUTO_THRESHOLD_ENABLED = False
    
    with patch("aegisx.ml.predict.Predictor._load_latest_model", return_value=(MagicMock(), {})):
        eng1 = TradingEngine(SimBroker(), mode="sim", force=True)
    assert eng1.threshold == 0.6
    
    # CLI Override
    settings.ENTRY_THRESHOLD_OVERRIDE = 0.35
    with patch("aegisx.ml.predict.Predictor._load_latest_model", return_value=(MagicMock(), {})):
        eng2 = TradingEngine(SimBroker(), mode="sim", force=True)
    assert eng2.threshold == 0.35
    
    settings.ENTRY_THRESHOLD_OVERRIDE = None

def test_expectancy_warmup(mock_engine):
    """Test warm-up logic for expectancy."""
    settings.EXPECTANCY_ENABLED = True
    settings.MIN_TRADES_FOR_EXPECTANCY = 5
    settings.MIN_EXPECTANCY_R = 10.0
    
    class MockRow(dict):
        @property
        def name(self): return "2025-01-01"

    row = MockRow({'close': 100, 'atr': 1})
    mock_engine.broker.get_balance = MagicMock(return_value=1000)
    
    # Case 1: 0 trades -> Should PASS
    passed = mock_engine._enter_trade(row, prob=0.5)
    assert passed == True
    
    # Case 2: 5 trades -> Should FAIL
    mock_engine.broker.trades = [{'type': 'SELL'}] * 5
    
    passed = mock_engine._enter_trade(row, prob=0.5)
    assert passed == False
    assert mock_engine.diagnostics['blocked_expectancy_count'] == 1