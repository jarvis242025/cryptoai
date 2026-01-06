from enum import Enum

class EngineState(Enum):
    IDLE = "IDLE"
    ARMED = "ARMED"      # Ready to fire (signal pending confirmation)
    IN_TRADE = "IN_TRADE"
    COOLDOWN = "COOLDOWN"
    HALTED = "HALTED"    # Killed by risk gate
