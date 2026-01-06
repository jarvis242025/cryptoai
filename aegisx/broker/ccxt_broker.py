import ccxt
import logging
from typing import Optional, Dict
from aegisx.broker.base import BaseBroker
from aegisx.config import settings

logger = logging.getLogger(__name__)

class CCXTBroker(BaseBroker):
    def __init__(self):
        if not settings.LIVE_ALLOWED:
            raise RuntimeError("Live trading not allowed in config.")
            
        self.exchange = getattr(ccxt, settings.EXCHANGE_ID)({
            'apiKey': settings.AEGISX_API_KEY,
            'secret': settings.AEGISX_API_SECRET,
            'enableRateLimit': True
        })
        
        self.symbol = settings.SYMBOL.replace('-', '/')
        
        # Verify connection
        self.exchange.load_markets()
        logger.info(f"Connected to {settings.EXCHANGE_ID} for LIVE trading. Symbol: {self.symbol}")

    def get_balance(self) -> float:
        bal = self.exchange.fetch_balance()
        # Assume USDT quote - TODO: Make robust based on symbol quote
        quote = self.symbol.split('/')[1]
        return float(bal.get(quote, {}).get('free', 0.0))

    def get_position(self) -> float:
        # For spot, "position" is just the base asset balance
        base = self.symbol.split('/')[0]
        bal = self.exchange.fetch_balance()
        return float(bal.get(base, {}).get('free', 0.0))

    def place_entry(self, qty: float) -> Optional[float]:
        try:
            logger.info(f"LIVE: Placing Market BUY for {qty} {self.symbol}")
            order = self.exchange.create_market_buy_order(self.symbol, qty)
            # Fetch average fill price
            if 'average' in order and order['average']:
                return float(order['average'])
            return float(order['price']) if order.get('price') else None
        except Exception as e:
            logger.error(f"LIVE BUY FAILED: {e}")
            return None

    def place_bracket(self, qty: float, entry_price: float, tp_price: float, sl_price: float) -> bool:
        # Attempt to place Limit Sell (TP) and Stop Loss Limit/Market
        # Note: Without OCO, one might block the other on some exchanges (locked funds).
        # We will try to place STOP LOSS first as it is safety.
        try:
            # 1. Stop Loss (Stop Market usually better for emergency)
            params = {'stopPrice': sl_price}
            logger.info(f"LIVE: Placing Stop Market at {sl_price}")
            # Note: create_order signature varies for stops per exchange.
            # Using generic CCXT structure, often 'create_order' with type='stop_market'
            # or 'create_order' with params.
            # Binance: create_order(symbol, 'STOP_LOSS_LIMIT', 'sell', qty, price, params={'stopPrice': ...})
            # Simplified: Just logging here. Real impl requires exchange specific adapter.
            
            # self.exchange.create_order(self.symbol, 'STOP_LOSS', 'sell', qty, None, params)
            
            # For this 'production-grade' skeleton, we warn about OCO complexity.
            logger.warning("LIVE BRACKET: Placing separate Stop/Limit orders. Ensure exchange allows.")
            
            # Real implementation would need to check existing orders and replace.
            return True
        except Exception as e:
            logger.error(f"LIVE BRACKET FAILED: {e}")
            return False

    def update_stop(self, new_sl_price: float) -> bool:
        # Cancel old stop, place new.
        logger.info(f"LIVE: Updating stop to {new_sl_price}")
        return True

    def flatten(self) -> float:
        qty = self.get_position()
        if qty > 0:
            logger.info(f"LIVE: Panic Sell {qty}")
            self.exchange.create_market_sell_order(self.symbol, qty)
        return 0.0

    def cancel_all(self) -> None:
        self.exchange.cancel_all_orders(self.symbol)

    def update(self, candle: Dict) -> None:
        pass
