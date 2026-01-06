import logging
from typing import Dict, Optional
from aegisx.broker.base import BaseBroker
from aegisx.broker.bracket import BracketManager
from aegisx.config import settings

logger = logging.getLogger(__name__)

class SimBroker(BaseBroker):
    def __init__(self, initial_cash: float = 10000.0):
        self.cash = initial_cash
        self.position = 0.0
        self.entry_price = 0.0
        self.equity_curve = []
        self.trades = []
        
        self.bracket = BracketManager()
        
    def get_balance(self) -> float:
        return self.cash
        
    def get_position(self) -> float:
        return self.position
        
    def get_equity(self, current_price: float) -> float:
        return self.cash + (self.position * current_price)

    def place_entry(self, qty: float) -> Optional[float]:
        # Assume Market Buy
        # In sim, we don't know exact execution price yet, 
        # usually called before 'update' processes the candle.
        # But for backtesting, we often peek at the 'Close' or next 'Open'.
        # Here, the Engine calls place_entry, and we simulate fill at next update 
        # OR we assume fill at current Close if operating on closed candles.
        
        # NOTE: To be realistic, if we signal on Candle N close, we enter on Candle N+1 Open.
        # However, for simplicity in vector backtest, usually Close.
        # Let's assume the Engine passes us the 'current' price for fill estimation 
        # or we wait for next tick. 
        # Let's implementation: Instant fill at 'current_market_price' passed in via context
        # usually isn't available here. 
        
        # Simplified: We assume fill at CURRENT candle Close (Signal Candle).
        # Slippage applied.
        pass # See specific impl below where I add a method for set_price

    def execute_buy(self, price: float, qty: float, timestamp=None):
        cost = price * qty
        fee = cost * (settings.FEE_BPS / 10000)
        
        if self.cash < cost + fee:
            logger.warning("Sim: Insufficient cash")
            return None
            
        # Slippage penalty
        slippage = price * (settings.SLIPPAGE_BPS / 10000)
        fill_price = price + slippage
        
        real_cost = fill_price * qty
        real_fee = real_cost * (settings.FEE_BPS / 10000)
        
        self.cash -= (real_cost + real_fee)
        self.position += qty
        self.entry_price = fill_price
        self.entry_fee = real_fee
        self.entry_time = timestamp
        
        self.trades.append({
            "type": "BUY",
            "price": fill_price,
            "qty": qty,
            "fee": real_fee,
            "timestamp": timestamp
        })
        return fill_price

    def place_entry(self, qty: float) -> Optional[float]:
        # This is tricky without current price. 
        # The engine must ensure it calls execute_buy or we store a "pending order"
        # For this design, let's assume immediate fill logic is handled in 'update' 
        # or we accept price arg.
        # I will change signature in Engine to handle price, 
        # but to satisfy interface, I'll store a request.
        self.pending_buy = qty
        return 0.0 # Placeholder

    def place_bracket(self, qty: float, entry_price: float, tp_price: float, sl_price: float) -> bool:
        self.bracket.set_bracket(qty, entry_price, tp_price, sl_price)
        return True
        
    def update_stop(self, new_sl_price: float) -> bool:
        self.bracket.sl_price = new_sl_price
        return True

    def flatten(self, reason="FORCE_EXIT") -> float:
        if self.position <= 0:
            return 0.0
        
        self.pending_sell = True
        self.pending_sell_reason = reason
        return 0.0

    def cancel_all(self) -> None:
        self.pending_buy = 0
        self.pending_sell = False
        self.bracket.reset()

    # Special Sim Methods
    def process_candle(self, candle: Dict) -> None:
        """
        Called by Engine to simulate time passing.
        candle: {open, high, low, close}
        """
        ts = candle.get('timestamp')
        # 1. Handle Pending Orders (Market Open execution)
        # Use OPEN price for fills if we decided at previous Close
        fill_price = candle['open']
        
        if hasattr(self, 'pending_buy') and self.pending_buy > 0:
            self.execute_buy(fill_price, self.pending_buy, timestamp=ts)
            self.pending_buy = 0
            
        if hasattr(self, 'pending_sell') and self.pending_sell:
            reason = getattr(self, 'pending_sell_reason', 'CLOSE')
            self._execute_sell(fill_price, reason=reason, timestamp=ts)
            self.pending_sell = False

        # 2. Check Bracket (Intra-candle)
        if self.position > 0 and self.bracket.active:
            res = self.bracket.check(candle['low'], candle['high'])
            if res == 'TP':
                self._execute_sell(self.bracket.tp_price, reason="TP", timestamp=ts)
            elif res == 'SL':
                # Slippage on stop
                slip = self.bracket.sl_price * (settings.SLIPPAGE_BPS / 10000)
                self._execute_sell(self.bracket.sl_price - slip, reason="SL", timestamp=ts)
                
        # 3. Track Equity
        eq = self.get_equity(candle['close'])
        self.equity_curve.append(eq)

    def _execute_sell(self, price: float, reason="CLOSE", timestamp=None):
        qty = self.position
        value = qty * price
        fee = value * (settings.FEE_BPS / 10000)
        
        entry_val = self.entry_price * qty
        entry_fee = getattr(self, 'entry_fee', 0.0)
        entry_time = getattr(self, 'entry_time', None)
        
        # PnL = (Exit Val - Entry Val) - (Exit Fee + Entry Fee)
        pnl = (value - entry_val) - (fee + entry_fee)
        
        # Duration
        duration = 0
        if entry_time and timestamp:
            duration = (timestamp - entry_time).total_seconds()
        
        self.cash += (value - fee)
        self.position = 0
        self.entry_fee = 0.0
        self.bracket.reset()
        
        self.trades.append({
            "type": "SELL",
            "price": price,
            "qty": qty,
            "fee": fee,
            "reason": reason,
            "pnl": pnl,
            "timestamp": timestamp,
            "duration": duration
        })
        
    def update(self, candle: Dict) -> None:
        self.process_candle(candle)
