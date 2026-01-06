from abc import ABC, abstractmethod
from typing import Dict, Optional

class BaseBroker(ABC):
    @abstractmethod
    def get_balance(self) -> float:
        pass
        
    @abstractmethod
    def get_position(self) -> float:
        pass
        
    @abstractmethod
    def place_entry(self, qty: float) -> Optional[float]:
        """Returns avg fill price or None"""
        pass
        
    @abstractmethod
    def place_bracket(self, qty: float, entry_price: float, tp_price: float, sl_price: float) -> bool:
        pass
        
    @abstractmethod
    def update_stop(self, new_sl_price: float) -> bool:
        pass
        
    @abstractmethod
    def flatten(self) -> float:
        """Close all positions. Returns exit price."""
        pass
        
    @abstractmethod
    def cancel_all(self) -> None:
        pass
        
    @abstractmethod
    def update(self, candle: Dict) -> None:
        """Called every tick/candle for sim updates"""
        pass
