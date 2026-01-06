import argparse
import logging
import sys
import time
import pandas as pd
from aegisx.config import settings
from aegisx.ml.train import train_model
from aegisx.engine.engine import TradingEngine
from aegisx.broker.sim import SimBroker
from aegisx.broker.ccxt_broker import CCXTBroker
from aegisx.data.ohlcv import OHLCVFetcher

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("CLI")

def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--exchange-id", type=str, help="Exchange ID (e.g. binanceus)")
    parser.add_argument("--symbol", type=str, help="Symbol (e.g. BTC/USDT)")
    parser.add_argument("--timeframe", type=str, help="Timeframe (e.g. 15m)")
    parser.add_argument("--max-rows", type=int, help="Max history rows to fetch")
    parser.add_argument("--fetch-limit", type=int, help="Rows per API call")
    parser.add_argument("--start-date", type=str, help="UTC start date YYYY-MM-DD; if provided, overrides --days")
    parser.add_argument("--force", action="store_true", help="Force execution despite model validation errors")

def apply_args(args):
    if args.exchange_id:
        settings.EXCHANGE_ID = args.exchange_id
    if args.symbol:
        settings.SYMBOL = args.symbol
    if args.timeframe:
        settings.TIMEFRAME = args.timeframe
    if args.max_rows:
        settings.MAX_OHLCV_ROWS = args.max_rows
    if args.fetch_limit:
        settings.FETCH_LIMIT = args.fetch_limit

def run_live_loop(engine: TradingEngine):
    logger.info(f"Starting Loop ({engine.mode})...")
    while True:
        try:
            # In live/paper, we fetch recent candles
            # We need enough for feature calc
            df = engine.fetcher.fetch_recent(lookback_days=2) 
            if df.empty:
                logger.warning("No data fetched.")
                time.sleep(10)
                continue

            last_row = df.iloc[-1]
            
            # TODO: Robust new-candle check (compare timestamps)
            engine.tick(last_row)
            
            time.sleep(60) # Wait 1 min
        except KeyboardInterrupt:
            logger.info("Stopping...")
            break
        except Exception as e:
            logger.error(f"Loop Error: {e}")
            time.sleep(60)

def main():
    parser = argparse.ArgumentParser(description="Aegis-X CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # TRAIN
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--days", type=int, default=365)
    train_parser.add_argument("--threshold-policy", type=str, choices=["max_precision", "target_rate"], default="max_precision", help="Threshold selection policy")
    train_parser.add_argument("--target-signals-per-day", type=float, default=2.0, help="Target signals/day for target_rate policy")
    add_common_args(train_parser)
    
    # SIM (Backtest)
    sim_parser = subparsers.add_parser("sim")
    sim_parser.add_argument("--days", type=int, default=30)
    sim_parser.add_argument("--threshold", type=float, help="Override entry threshold")
    add_common_args(sim_parser)

    # PAPER (Live Data, Sim Broker)
    paper_parser = subparsers.add_parser("paper")
    add_common_args(paper_parser)
    
    # LIVE (Real Money)
    live_parser = subparsers.add_parser("live")
    live_parser.add_argument("--i-understand-this-can-lose-money", action="store_true")
    add_common_args(live_parser)
    
    args = parser.parse_args()
    
    # Apply global overrides
    if hasattr(args, 'exchange_id'):
        apply_args(args)
    
    if settings.FIB_ENABLED:
        logger.info(f"Fibonacci Enabled: Gate={settings.FIB_GATE_ENABLED}, DynamicBrackets={settings.FIB_USE_DYNAMIC_BRACKETS}")
    
    if args.command == "train":
        start_date = getattr(args, "start_date", None)
        train_model(
            days=args.days, 
            start_date=start_date,
            threshold_policy=args.threshold_policy,
            target_signals_per_day=args.target_signals_per_day
        )
        
    elif args.command == "sim":
        fetcher = OHLCVFetcher()
        if args.threshold:
            settings.ENTRY_THRESHOLD_OVERRIDE = args.threshold
        start_date_str = getattr(args, "start_date", None)
        
        if start_date_str:
            from datetime import datetime
            try:
                dt_start = datetime.strptime(start_date_str, "%Y-%m-%d")
                df = fetcher.fetch_from(dt_start)
            except ValueError:
                logger.error("Invalid start-date format. Use YYYY-MM-DD.")
                return
        else:
            df = fetcher.fetch_recent(lookback_days=args.days)
        
        if df.empty:
            logger.error("No data fetched for simulation.")
            return
        
        broker = SimBroker()
        engine = TradingEngine(broker, mode="sim", force=args.force)
        engine.run_simulation(df)
        
        # Report
        total_trades = len([t for t in broker.trades if t['type'] == 'SELL'])
        
        # Calculate stats
        exits = [t for t in broker.trades if t['type'] == 'SELL']
        
        wins = len([t for t in exits if t.get('pnl', 0) > 0])
        losses = len([t for t in exits if t.get('pnl', 0) <= 0])
        
        total_pnl = sum(t.get('pnl', 0) for t in exits)
        total_fees = sum(t.get('fee', 0) for t in broker.trades) # Buy + Sell fees
        
        durations = [t.get('duration', 0) for t in exits if t.get('duration', 0) > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        print("\n--- SIMULATION RESULTS ---")
        print(f"Final Balance: {float(broker.get_balance()):.2f}")
        print(f"Net PnL: {total_pnl:.2f}")
        print(f"Total Fees: {total_fees:.2f}")
        print(f"Trades: {total_trades}")
        print(f"Wins: {wins} ({wins/total_trades*100 if total_trades else 0:.1f}%)")
        print(f"Losses: {losses} ({losses/total_trades*100 if total_trades else 0:.1f}%)")
        print(f"Avg Duration: {avg_duration/60:.1f} min")
        
        print("\nExit Reasons:")
        reasons = {}
        for t in exits:
            r = t.get('reason', 'UNKNOWN')
            reasons[r] = reasons.get(r, 0) + 1
        for r, c in reasons.items():
            print(f"  {r}: {c}")

    elif args.command == "paper":
        # SimBroker but Live Loop
        broker = SimBroker()
        engine = TradingEngine(broker, mode="paper", force=args.force) # 'paper' mode logic same as live but safe
        # We need to ensure engine handles 'paper' mode correctly in tick (it mostly checks != sim for news)
        # Check engine.py:
        # if self.mode != "sim":
        #    risk_score = self.news.check_risk()
        # So 'paper' will check news. Good.
        
        run_live_loop(engine)
        
    elif args.command == "live":
        if not args.i_understand_this_can_lose_money:
            logger.error("You must confirm you understand risks.")
            return
            
        if not settings.LIVE_ALLOWED:
            logger.error("LIVE_ALLOWED is False in config.")
            return
            
        broker = CCXTBroker()
        engine = TradingEngine(broker, mode="live", force=args.force)
        run_live_loop(engine)

if __name__ == "__main__":
    main()