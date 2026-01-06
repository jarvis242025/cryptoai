from aegisx.config import settings

class BracketManager:
    """
    Manages TP/SL logic.
    Used internally by SimBroker, and by Engine for soft-stops in Live mode 
    if exchange-side brackets aren't atomic.
    """
    def __init__(self):
        self.entry_price = 0.0
        self.tp_price = 0.0
        self.sl_price = 0.0
        self.qty = 0.0
        self.active = False
        self.highest_price = 0.0
        
    def set_bracket(self, qty, entry, tp, sl):
        self.qty = qty
        self.entry_price = entry
        self.tp_price = tp
        self.sl_price = sl
        self.active = True
        self.highest_price = entry
        
    def check(self, low: float, high: float) -> str:
        """
        Returns: 'TP', 'SL', or None
        """
        if not self.active:
            return None
            
        # Update ratchet trigger tracker
        if high > self.highest_price:
            self.highest_price = high
            
        # Check Ratchet/Breakeven
        # If price moved X% in favor, move SL to Entry + Fees
        gain_pct = (self.highest_price - self.entry_price) / self.entry_price
        if gain_pct >= settings.RATCHET_TRIGGER:
            # Move SL to breakeven (entry * 1.002 roughly for fees)
            be_price = self.entry_price * (1 + settings.FEE_BPS/10000 * 2.5) 
            if be_price > self.sl_price:
                self.sl_price = be_price

        # Check Exits
        # Assume if both hit, SL hit first (pessimistic) unless we have tick data
        if low <= self.sl_price:
            return 'SL'
        if high >= self.tp_price:
            return 'TP'
            
        return None
        
    def reset(self):
        self.active = False
        self.qty = 0
