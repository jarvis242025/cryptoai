import pytest
from aegisx.broker.sim import SimBroker
from aegisx.config import settings

def test_sim_accounting_buy():
    settings.FEE_BPS = 10 # 0.1%
    settings.SLIPPAGE_BPS = 0 # Disable for simpler math
    
    broker = SimBroker(initial_cash=10000.0)
    
    # Buy 1 BTC at 1000
    price = 1000.0
    qty = 1.0
    
    broker.execute_buy(price, qty)
    
    # Cost = 1000
    # Fee = 1000 * 0.001 = 1.0
    # Cash should be 10000 - 1001 = 8999
    
    assert broker.cash == 8999.0
    assert broker.position == 1.0
    assert broker.entry_price == 1000.0

def test_sim_bracket_exit():
    settings.FEE_BPS = 0
    settings.SLIPPAGE_BPS = 0
    
    broker = SimBroker(initial_cash=1000.0)
    broker.position = 1.0
    broker.entry_price = 100.0
    broker.cash = 0.0
    
    # TP at 110, SL at 90
    broker.place_bracket(1.0, 100.0, 110.0, 90.0)
    
    # Candle hits TP
    candle = {'open': 105, 'high': 115, 'low': 105, 'close': 112}
    broker.process_candle(candle)
    
    assert broker.position == 0.0
    assert broker.cash == 110.0 # Sold at 110
    assert len(broker.trades) == 1
    assert broker.trades[0]['reason'] == 'TP'
