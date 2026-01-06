import ccxt
import pandas as pd
import time
import logging
from datetime import datetime, timedelta
from typing import Optional
from aegisx.config import settings

logger = logging.getLogger(__name__)

class OHLCVFetcher:
    def __init__(self) -> None:
        try:
            exchange_cls = getattr(ccxt, settings.EXCHANGE_ID)
        except AttributeError:
            raise ValueError(
                f"Invalid EXCHANGE_ID '{settings.EXCHANGE_ID}'. "
                "Please set AEGISX_EXCHANGE_ID environment variable or use --exchange-id CLI flag."
            )
        
        self.exchange = exchange_cls({
            "enableRateLimit": True, 
            "options": {"defaultType": "spot"}
        })
        logger.info(f"Using Exchange: {settings.EXCHANGE_ID}")

    def fetch_data(self, limit: int = 1000, since: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch single batch of OHLCV data from exchange.
        """
        logger.debug(f"Fetching batch: limit={limit}, since={since}")
        
        since_ts = int(since.timestamp() * 1000) if since else None
        
        try:
            symbol_norm = settings.SYMBOL.replace('-', '/')
            ohlcv = self.exchange.fetch_ohlcv(
                symbol=symbol_norm,
                timeframe=settings.TIMEFRAME,
                since=since_ts,
                limit=limit
            )
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Don't set index yet to allow concatenation easier, or set it and reset later.
            # Existing code sets index. Let's keep it consistent but handle duplicates in caller.
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                df.sort_index(inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV: {e}")
            raise

    def fetch_recent(self, lookback_days: int = 30) -> pd.DataFrame:
        """
        Fetch historical data with pagination.
        """
        now = datetime.utcnow()
        start_date = now - timedelta(days=lookback_days)
        return self.fetch_from(start_date)

    def fetch_from(self, start_date: datetime) -> pd.DataFrame:
        """
        Fetch historical data from start_date until now with pagination.
        """
        now = datetime.utcnow()
        current_since = start_date
        all_dfs = []
        total_rows = 0
        
        logger.info(f"Fetching data starting {start_date}...")
        
        # Convert timeframe to ms for increment
        tf_secs = self.exchange.parse_timeframe(settings.TIMEFRAME)
        tf_ms = tf_secs * 1000
        
        while total_rows < settings.MAX_OHLCV_ROWS:
            if current_since >= now:
                break
                
            batch = self.fetch_data(limit=settings.FETCH_LIMIT, since=current_since)
            
            if batch.empty:
                logger.info("Exchange returned empty batch. Stopping.")
                break
                
            # Check for stagnation
            last_time = batch.index[-1]
            last_ts_ms = int(last_time.timestamp() * 1000)
            
            all_dfs.append(batch)
            rows_in_batch = len(batch)
            total_rows += rows_in_batch
            
            logger.info(f"Fetched {rows_in_batch} rows. Total: {total_rows}. Covered up to {last_time}")
            
            # Prepare next since: last_time + 1 timeframe unit
            current_since = last_time + timedelta(milliseconds=tf_ms)
            
            # Safety break if we reached close to now
            if (now - last_time).total_seconds() < tf_secs:
                break
                
            time.sleep(self.exchange.rateLimit / 1000.0) # Respect rate limit
            
        if not all_dfs:
            return pd.DataFrame()
            
        final_df = pd.concat(all_dfs)
        final_df = final_df[~final_df.index.duplicated(keep='first')]
        final_df.sort_index(inplace=True)
        
        logger.info(f"Final Dataset: {len(final_df)} rows from {final_df.index[0]} to {final_df.index[-1]}")
        return final_df
