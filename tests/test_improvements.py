import pytest
from aegisx.broker.sim import SimBroker
from aegisx.config import settings

def test_sim_anomaly_exit_tracking():
    broker = SimBroker(initial_cash=10000.0)
    broker.position = 1.0
    broker.entry_price = 100.0
    broker.entry_fee = 1.0
    broker.cash = 0.0 # All in
    
    # Force exit due to anomaly
    broker.flatten(reason="ANOMALY")
    
    # Process candle to execute
    candle = {'open': 95, 'high': 105, 'low': 90, 'close': 98, 'timestamp': 1000}
    broker.process_candle(candle)
    
    assert len(broker.trades) == 1
    trade = broker.trades[0]
    assert trade['type'] == 'SELL'
    assert trade['reason'] == 'ANOMALY'
    assert trade['price'] == 95 # Market Open fill
    
    # PnL check
    # Entry: 100 * 1 = 100. Fee 1.
    # Exit: 95 * 1 = 95. Fee = 95 * 0.006 (60bps) = 0.57
    # PnL = (95 - 100) - (0.57 + 1.0) = -5 - 1.57 = -6.57
    expected_pnl = (95 - 100) - (trade['fee'] + 1.0)
    assert trade['pnl'] == pytest.approx(expected_pnl, 0.01)
