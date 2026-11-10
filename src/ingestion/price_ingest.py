"""
Price bar ingestion using yfinance with timezone standardization to UTC.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class PriceIngestion:
    def __init__(self, cache_dir: Path, bar_interval: str = "1h", days_lookback: int = 14):
        self.cache_dir = cache_dir / "prices"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.bar_interval = bar_interval
        self.days_lookback = days_lookback

    def fetch_bars(
        self,
        ticker: str,
        exchange_tz: str = "UTC",
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch intraday (1h) price bars for a ticker, convert to UTC index, and cache as parquet.
        """
        cache_file = self.cache_dir / f"{ticker}_{self.bar_interval}.parquet"

        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                logger.info(f"Loaded {len(df)} price bars from cache for {ticker}")
                return df
            except Exception as e:
                logger.warning(f"Could not read cache for {ticker}: {e}. Refetching.")

        logger.info(f"Downloading {self.days_lookback}d of {self.bar_interval} bars for {ticker}...")
        raw_df = yf.download(
            ticker,
            period=f"{self.days_lookback}d",
            interval=self.bar_interval,
            progress=False,
            auto_adjust=True,
        )

        if raw_df.empty:
            raise ValueError(f"No price data retrieved for ticker {ticker}")

        # Flatten multi-index columns if returned by yfinance
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = [col[0] for col in raw_df.columns]

        df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()

        # Ensure index is timezone-aware and convert to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize(exchange_tz).tz_convert(timezone.utc)
        else:
            df.index = df.index.tz_convert(timezone.utc)

        df.index.name = "timestamp_utc"

        # Drop any empty or invalid bars
        df = df.dropna().sort_index()

        # Cache locally
        df.to_parquet(cache_file)
        logger.info(f"Cached {len(df)} price bars for {ticker} -> {cache_file}")
        return df

