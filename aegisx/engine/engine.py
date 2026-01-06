import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional

from aegisx.config import settings
from aegisx.engine.state import EngineState
from aegisx.broker.base import BaseBroker
from aegisx.broker.sim import SimBroker
from aegisx.ml.predict import Predictor
from aegisx.features.build_features import build_features
from aegisx.risk.news_radar import NewsRadar
from aegisx.risk.anomaly import AnomalyDetector
from aegisx.data.ohlcv import OHLCVFetcher
from aegisx.ta.fibonacci import get_fib_levels

logger = logging.getLogger(__name__)

class TradingEngine:
    def __init__(self, broker: BaseBroker, mode: str = "sim", force: bool = False):
        self.broker = broker
        self.mode = mode
        self.state = EngineState.IDLE
        
        self.fetcher = OHLCVFetcher()
        self.predictor = Predictor()
        
        # Validate Model
        self.predictor.validate_model(force=force)
        
        self.news = NewsRadar()
        self.anomaly = AnomalyDetector()
        
        # Threshold Resolution
        self.threshold = settings.ENTRY_THRESHOLD
        source = "config"
        
        if settings.AUTO_THRESHOLD_ENABLED:
            # Predictor already loaded metadata in its __init__
            rec_thr = getattr(self.predictor, 'recommended_threshold', None)
            if rec_thr is not None:
                self.threshold = rec_thr
                source = "model (auto)"
        
        # Override (highest precedence)
        if settings.ENTRY_THRESHOLD_OVERRIDE is not None:
            self.threshold = settings.ENTRY_THRESHOLD_OVERRIDE
            source = "cli/override"

        logger.info(f"Using entry threshold: {self.threshold:.4f} (source: {source})")
        
        # Fee/Cost Calc
        self.maker_fee = settings.MAKER_FEE_BPS / 10000
        self.taker_fee = settings.TAKER_FEE_BPS / 10000
        self.est_spread = settings.EST_SPREAD_BPS / 10000
        self.est_slip = settings.EST_SLIPPAGE_BPS / 10000
        
        # cost_bps = (2 * MAKER_FEE_BPS) + EST_SPREAD_BPS + EST_SLIPPAGE_BPS (per prompt)
        # Note: Prompt specified BPS ints in formula, so convert to float pct at end or use bps consistently.
        # Let's use pct for final value.
        cost_bps = (2 * settings.MAKER_FEE_BPS) + settings.EST_SPREAD_BPS + settings.EST_SLIPPAGE_BPS
        self.cost_pct = cost_bps / 10000
        
        # Effective TP min
        self.min_tp_pct = max(settings.TP_PCT, 2.0 * self.cost_pct)
        
        logger.info(f"Fee Model (BPS): Maker={settings.MAKER_FEE_BPS}, Taker={settings.TAKER_FEE_BPS}, Spread={settings.EST_SPREAD_BPS}, Slip={settings.EST_SLIPPAGE_BPS}")
        logger.info(f"Est Round-Trip Cost: {cost_bps} bps ({self.cost_pct:.4f})")
        logger.info(f"Minimum Effective TP: {self.min_tp_pct:.4f} (Config TP={settings.TP_PCT})")
        
        # Diagnostics
        self.diagnostics = {
            "total_ticks": 0,
            "trend_ok_count": 0,
            "prob_pass_count": 0,
            "candidate_count": 0,
            "entries_count": 0,
            "blocked_anomaly_count": 0,
            "blocked_news_count": 0,
            "blocked_fib_count": 0,
            "blocked_hurst_count": 0,
            "blocked_expectancy_count": 0,
            "all_probs": [],
            "hurst_values": []
        }

        # Fit anomaly detector if we can fetch history
        if mode == "sim":
             # In sim, we fit on past data relative to current sim step? 
             # Or pre-fit. For simplicity, we assume pre-fit or fit on start.
             pass
        else:
             logger.info("Warming up Anomaly Detector...")
             hist = self.fetcher.fetch_recent(30)
             hist = build_features(hist)
             self.anomaly.fit(hist)

    def run_simulation(self, df: pd.DataFrame):
        """
        Run loop over historical dataframe.
        """
        logger.info(f"Starting Simulation on {len(df)} candles...")
        
        # Pre-calculate features for speed
        df = build_features(df)
        
        # Train anomaly detector on first portion or rolling? 
        # We'll just fit on the whole dataset for this MVP to define "normal" regime
        # Real-world: Rolling fit.
        self.anomaly.fit(df)
        
        for i in range(50, len(df)):
            # Sim "current" candle is i
            # But we make decision based on i (Close) to enter at i+1 (Open)
            # Broker update handles i+1 Open fill if we placed order at i.
            
            # Context: We are AT time i.
            current_row = df.iloc[i]
            
            # 1. Update Broker with CURRENT candle (to process pending fills/exits from previous tick)
            # Convert row to dict
            candle = {
                'timestamp': current_row.name,
                'open': current_row['open'],
                'high': current_row['high'],
                'low': current_row['low'],
                'close': current_row['close']
            }
            self.broker.update(candle)
            
            # 2. State Management
            pos = self.broker.get_position()
            if pos > 0:
                self.state = EngineState.IN_TRADE
            elif self.state == EngineState.IN_TRADE and pos <= 0:
                self.state = EngineState.IDLE
                
            # 3. Logic
            self.tick(current_row)

        # End
        self.broker.flatten()
        logger.info("Simulation Complete.")
        
        # Diagnostics Output
        d = self.diagnostics
        total = d["total_ticks"]
        probs = d["all_probs"]
        
        print("\n--- GATE DIAGNOSTICS ---")
        print(f"Total Ticks: {total}")
        if total > 0:
            print(f"Trend OK: {d['trend_ok_count']} ({d['trend_ok_count']/total*100:.1f}%)")
            print(f"Prob Pass: {d['prob_pass_count']} ({d['prob_pass_count']/total*100:.1f}%)")
            print(f"Candidates: {d['candidate_count']} (Trend + Prob)")
            
            if d['candidate_count'] > 0:
                print("\nFunnel (Blocks from Candidates):")
                print(f"  Blocked by Anomaly: {d['blocked_anomaly_count']}")
                print(f"  Blocked by News: {d['blocked_news_count']}")
                print(f"  Blocked by Fib: {d['blocked_fib_count']}")
                print(f"  Blocked by Hurst: {d['blocked_hurst_count']}")
                print(f"  Blocked by Expectancy: {d['blocked_expectancy_count']}")
                print(f"  ENTRIES: {d['entries_count']}")
            
            if d['hurst_values']:
                h_arr = np.array(d['hurst_values'])
                below = (h_arr < settings.HURST_MIN).sum()
                print(f"\nHurst Stats (min={settings.HURST_MIN}):")
                print(f"  Min: {h_arr.min():.3f}  Max: {h_arr.max():.3f}  Mean: {h_arr.mean():.3f}")
                print(f"  p50: {np.percentile(h_arr, 50):.3f}  p90: {np.percentile(h_arr, 90):.3f}")
                print(f"  Below Limit: {below} ({below/len(h_arr)*100:.1f}%)")
                
            if probs:
                p_arr = np.array(probs)
                print("\nProbability Stats:")
                print(f"  Min: {p_arr.min():.4f}  Max: {p_arr.max():.4f}  Mean: {p_arr.mean():.4f}")
                print(f"  p50: {np.percentile(p_arr, 50):.4f}")
                print(f"  p90: {np.percentile(p_arr, 90):.4f}")
                print(f"  p99: {np.percentile(p_arr, 99):.4f}")
        print("------------------------\n")
        
    def tick(self, row: pd.Series):
        """
        Single tick logic.
        """
        self.diagnostics["total_ticks"] += 1
        
        # Anomaly Logic (Always run for potential exit)
        is_anomaly = self.anomaly.is_anomaly(row)

        if self.state == EngineState.HALTED:
             return # Halt requires manual or cooldown reset (not implemented)

        if self.state == EngineState.IN_TRADE:
             # ... existing exit logic ...
             # Severe Anomaly Exit Logic
             # Check required features exist (safe access)
             ret1 = row.get('returns_1', 0.0)
             vol_z = row.get('vol_z', 0.0)
             
             is_crash = ret1 <= settings.ANOMALY_KILL_RETURN
             is_vol_dump = is_anomaly and (vol_z >= settings.ANOMALY_KILL_VOL_Z) and (ret1 < 0)
             
             if is_crash or is_vol_dump:
                 logger.warning(f"Severe Anomaly Exit: Ret1={ret1:.4f}, VolZ={vol_z:.2f}. Exiting.")
                 self.broker.flatten(reason="ANOMALY_KILL")
                 return # Exit processed
                 
        if self.state == EngineState.IDLE:
            # 1. Trend Filter
            ema_50 = row.get('ema_50', 0.0)
            trend_ok = row['close'] > ema_50 if ema_50 > 0 else False
            if trend_ok:
                self.diagnostics["trend_ok_count"] += 1
            
            # 2. Probability
            prob = self.predictor.predict_probability(row)
            self.diagnostics["all_probs"].append(prob)
            
            # Dynamic Threshold (Confluence adjustment)
            # We calculate this first to see if prob passes
            required_threshold = self.threshold
            fib_confluence = False
            
            if settings.FIB_CONFLUENCE_ENABLED:
                in_golden = row.get('fib_in_golden', 0)
                if in_golden:
                    fib_confluence = True
                    required_threshold = max(0.01, required_threshold - settings.FIB_THRESHOLD_DISCOUNT)
            
            prob_pass = prob > required_threshold
            if prob_pass:
                self.diagnostics["prob_pass_count"] += 1
                
            # 3. Candidate?
            if trend_ok and prob_pass:
                self.diagnostics["candidate_count"] += 1
                
                # --- GATE FUNNEL (Only check if candidate) ---
                
                # A) Anomaly Gate
                if is_anomaly:
                    self.diagnostics["blocked_anomaly_count"] += 1
                    return
                
                # B) News Gate
                if self.mode != "sim":
                    risk_score = self.news.check_risk()
                    if risk_score > settings.NEWS_RISK_THRESHOLD:
                         self.diagnostics["blocked_news_count"] += 1
                         return
                
                # C) Fib Gate (if enabled as HARD gate)
                # Note: settings.FIB_GATE_ENABLED is usually False (confluence only)
                if settings.FIB_GATE_ENABLED:
                    in_golden = row.get('fib_in_golden', 0)
                    if not in_golden:
                        self.diagnostics["blocked_fib_count"] += 1
                        return
                        
                # D) Hurst Gate
                if settings.HURST_ENABLED:
                    hurst_val = row.get('hurst', 0.5)
                    self.diagnostics["hurst_values"].append(hurst_val)
                    if hurst_val < settings.HURST_MIN:
                        self.diagnostics["blocked_hurst_count"] += 1
                        return
                
                # E) Expectancy Gate
                # This depends on TP/SL (b).
                # We do this inside _enter_trade to avoid duplicating bracket logic?
                # Or we calculate 'b' here?
                # The instruction says: "Only if candidate is true, apply gates (anomaly/news/fib/hurst/expectancy) and count them."
                # We'll delegate expectancy counting to _enter_trade but handle return value to know if blocked.
                
                entered = self._enter_trade(row, prob)
                if entered:
                    self.diagnostics["entries_count"] += 1
                    self.state = EngineState.IN_TRADE

    def _enter_trade(self, row: pd.Series, prob: float = 0.5) -> bool:
        price = row['close']
        
        # Sizing Base
        balance = self.broker.get_balance()
        risk_pct = settings.RISK_PER_TRADE
        
        # Determine Stops First (needed for R:R and Expectancy)
        sl_pct = settings.SL_PCT
        tp_pct = self.min_tp_pct
        
        # Dynamic Fib Brackets
        if settings.FIB_USE_DYNAMIC_BRACKETS:
            # Need swing levels
            sh = row.get('fib_swing_high')
            sl_val = row.get('fib_swing_low')
            impulse_up = row.get('fib_impulse_up')
            
            if pd.notna(sh) and pd.notna(sl_val):
                direction = 'UP' if impulse_up else 'DOWN'
                levels = get_fib_levels(sl_val, sh, direction)
                
                # Select TP
                best_tp = 0.0
                for ext in settings.FIB_TP_EXTENSIONS:
                    lvl_price = levels.get(f'ext_{ext}')
                    if lvl_price:
                        potential_return = (lvl_price - price) / price
                        if potential_return >= self.min_tp_pct:
                            best_tp = lvl_price
                            break 
                
                if best_tp > 0:
                    tp_pct = (best_tp - price) / price
                    
                # Select SL
                sl_level = levels.get('sl_618')
                if sl_level and sl_level < price:
                    proposed_sl_pct = (price - sl_level) / price
                    if proposed_sl_pct >= settings.SL_PCT:
                        sl_pct = proposed_sl_pct
        
        elif settings.ATR_BRACKETS_ENABLED and 'atr' in row:
            atr = row['atr']
            if atr > 0:
                atr_pct = atr / price
                tp_pct = max(self.min_tp_pct, settings.ATR_TP_MULT * atr_pct)
                sl_pct = max(settings.SL_PCT, settings.ATR_SL_MULT * atr_pct)

        # Quant Trinity: Expectancy & Kelly
        # 1. Calculate Reward/Risk Ratio (b) including costs
        # Reward = TP_pct - cost
        # Risk = SL_pct + cost
        reward_net = tp_pct - self.cost_pct
        risk_net = sl_pct + self.cost_pct
        
        if risk_net <= 0: b = 0.0
        else: b = reward_net / risk_net
        
        # Trade Count for Warm-up
        # Note: broker.trades includes both BUY and SELL. A complete trade is usually 2 ops.
        # We want to know how many trades we've DONE.
        # Let's count exits? Or just entries?
        # SimBroker.trades is a list of dicts.
        n_trades = len([t for t in self.broker.trades if t['type'] == 'SELL'])
        
        # 2. Expectancy Check
        if settings.EXPECTANCY_ENABLED:
            if n_trades >= settings.MIN_TRADES_FOR_EXPECTANCY:
                expectancy_r = (prob * b) - (1.0 - prob)
                if expectancy_r < settings.MIN_EXPECTANCY_R:
                    # logger.info(f"Blocked by Expectancy: {expectancy_r:.2f} < {settings.MIN_EXPECTANCY_R}")
                    self.diagnostics["blocked_expectancy_count"] += 1
                    return False
            else:
                expectancy_r = 0.0 # Warm-up pass

        # 3. Kelly Sizing
        if settings.KELLY_ENABLED and b > 0:
            if n_trades >= settings.MIN_TRADES_FOR_KELLY:
                # Kelly Fraction f = p - q/b = p - (1-p)/b = (pb - 1 + p) / b = (p(b+1) - 1) / b
                # Full Kelly
                kelly_f = (prob * (b + 1) - 1) / b
                kelly_f = max(0.0, kelly_f)
                
                # Partial Kelly & Cap
                risk_pct = min(settings.KELLY_MAX_RISK, kelly_f * settings.KELLY_FRACTION)
                
                # If Kelly says 0 risk (edge not sufficient), we skip
                if risk_pct <= 0:
                    return False
            # Else fallback to default risk_pct
        else:
            risk_pct = settings.RISK_PER_TRADE

        # Stop distance logic
        risk_amt = balance * risk_pct
        dist = price * sl_pct
        
        # Qty = Risk / Distance
        if dist == 0: return False
        qty = risk_amt / dist
        
        # Execute
        logger.info(f"Entry: P={prob:.2f} E[R]={expectancy_r:.2f} Risk={risk_pct*100:.2f}% (Trades={n_trades})")
        self.broker.place_entry(qty)
        
        # Bracket
        tp = price * (1 + tp_pct)
        sl = price * (1 - sl_pct)
        self.broker.place_bracket(qty, price, tp, sl)
        
        return True
