import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Exchange
    EXCHANGE_ID: str = "coinbase"
    SYMBOL: str = "SOL-USD"
    TIMEFRAME: str = "15m"
    MAX_OHLCV_ROWS: int = 40000
    FETCH_LIMIT: int = 1000
    
    # Strategy
    TP_PCT: float = 0.07
    SL_PCT: float = 0.03
    RATCHET_TRIGGER: float = 0.03     # When to move stop to break-even (Ratchet)
    RISK_PER_TRADE: float = 0.01      # % of equity per trade

    # Quant Trinity: Prediction & Safety
    MODEL_TYPE: str = "rf"            # "rf" or "xgb"
    
    # XGBoost Params
    XGB_N_ESTIMATORS: int = 300
    XGB_MAX_DEPTH: int = 5
    XGB_LEARNING_RATE: float = 0.05
    XGB_SUBSAMPLE: float = 0.8
    XGB_COLSAMPLE_BYTREE: float = 0.8
    
    HURST_ENABLED: bool = True
    HURST_WINDOW: int = 200           # Rolling window size
    HURST_MIN: float = 0.52           # Regime filter (0.5=Random Walk, >0.5=Trend)
    
    KELLY_ENABLED: bool = True
    KELLY_FRACTION: float = 0.5       # Fractional Kelly safety
    KELLY_MAX_RISK: float = 0.02      # Cap absolute risk per trade
    
    EXPECTANCY_ENABLED: bool = True
    MIN_EXPECTANCY_R: float = 0.10    # Minimum expectany (R-multiple)
    
    # Fibonacci Confluence
    FIB_CONFLUENCE_ENABLED: bool = True
    FIB_THRESHOLD_DISCOUNT: float = 0.03 # Reduce threshold if Fib aligns
    
    # Fibonacci
    FIB_ENABLED: bool = False
    FIB_GATE_ENABLED: bool = False
    FIB_LOOKBACK_BARS: int = 200
    FIB_PIVOT_LEFT: int = 3
    FIB_PIVOT_RIGHT: int = 3
    FIB_GOLDEN_LOW: float = 0.50
    FIB_GOLDEN_HIGH: float = 0.618
    FIB_USE_DYNAMIC_BRACKETS: bool = False
    FIB_TP_EXTENSIONS: list[float] = [1.0, 1.272, 1.618]
    
    # Fees & Spread (Coinbase Advanced Tier defaults)
    FEE_BPS: float = 60.0             # SimBroker uses this (Taker default)
    MAKER_FEE_BPS: float = 40.0
    TAKER_FEE_BPS: float = 60.0
    EST_SPREAD_BPS: float = 10.0
    EST_SLIPPAGE_BPS: float = 10.0
    SLIPPAGE_BPS: float = 5.0         # Kept for compatibility if used elsewhere, but EST_SLIPPAGE_BPS preferred for calc
    
    # ATR Brackets
    ATR_BRACKETS_ENABLED: bool = True
    ATR_TP_MULT: float = 2.0
    ATR_SL_MULT: float = 1.0
    
    HORIZON_BARS: int = 96            # Max hold time (24h for 15m)
    MIN_PROB: float = 0.60            # ML probability threshold
    TREND_EMA_LEN: int = 50           # Trend filter
    ENTRY_THRESHOLD: float = 0.20     # Low fallback to ensure trades if model missing
    ENTRY_THRESHOLD_OVERRIDE: Optional[float] = None # CLI override

    # Threshold Selection
    MIN_VAL_SIGNALS: int = 1          # Auto-threshold target (deprecated name)
    MIN_SIGNALS: int = 1
    MIN_SIGNALS_TOTAL: int = 10       # Min total validation signals for target_rate policy
    TARGET_SIGNALS_PER_DAY: float = 0.25
    MIN_VALIDATION_PRECISION: float = 0.40
    AUTO_THRESHOLD_ENABLED: bool = True
    
    # Warm-up Rules
    MIN_TRADES_FOR_EXPECTANCY: int = 5
    MIN_TRADES_FOR_KELLY: int = 5
    
    # Anomaly Kill Switch
    ANOMALY_KILL_RETURN: float = -0.03
    ANOMALY_KILL_VOL_Z: float = 3.0
    
    # News
    NEWS_ENABLED: bool = True
    NEWS_COOLDOWN_MIN: int = 60
    NEWS_RISK_THRESHOLD: int = 50
    
    # Live
    LIVE_ALLOWED: bool = False
    AEGISX_API_KEY: str = ""
    AEGISX_API_SECRET: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()
