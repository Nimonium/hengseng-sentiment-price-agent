"""
Strict Zero-Lookahead Alignment Engine.
Aligns social posts and news articles to price bars without lookahead leakage.
A post published at timestamp t can strictly only influence bar features at or after t.
Overnight and weekend posts roll into the first bar of the next market open.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.sentiment.aggregator import SentimentAggregator

logger = logging.getLogger(__name__)


class FeatureAligner:
    def __init__(self, aggregator: Optional[SentimentAggregator] = None, half_life_bars: float = 6.0):
        self.aggregator = aggregator or SentimentAggregator()
        self.half_life_bars = half_life_bars

    def align_sentiment_to_bars(
        self,
        price_df: pd.DataFrame,
        social_df: pd.DataFrame,
        news_df: Optional[pd.DataFrame] = None,
        source_prefix: str = "social",
    ) -> pd.DataFrame:
        """
        Perform strict point-in-time causal join.
        For price bar k with timestamp T_k, only items with publication timestamp t in (T_{k-1}, T_k]
        are aggregated into bar k features.

        Under no circumstances can an item with t > T_k be included in bar k.
        Continuous sentiment state is maintained with an exponential half-life decay,
        ensuring sentiment carries forward across trading bars without dropping to zero.
        Overnight and weekend posts naturally roll into the first trading bar after market open.
        """
        df = price_df.copy().sort_index()
        bar_timestamps = df.index.to_series()

        n_bars = len(df)
        agg_scores = np.zeros(n_bars, dtype=np.float64)
        agg_mentions = np.zeros(n_bars, dtype=np.int64)
        agg_dispersion = np.zeros(n_bars, dtype=np.float64)
        agg_bull_bear = np.zeros(n_bars, dtype=np.float64)
        agg_engagement = np.zeros(n_bars, dtype=np.float64)

        if social_df is not None and not social_df.empty and "timestamp" in social_df.columns:
            s_df = social_df.copy()
            s_df["timestamp"] = pd.to_datetime(s_df["timestamp"], utc=True)
            s_df = s_df.sort_values("timestamp")

            for i in range(n_bars):
                curr_t = bar_timestamps.iloc[i]
                # For bar 0, allow 72h pre-market lookback so overnight/weekend posts roll into open bar
                prev_t = bar_timestamps.iloc[i - 1] if i > 0 else (curr_t - pd.Timedelta(hours=72))

                # Strictly causal filter: prev_t < timestamp <= curr_t
                mask = (s_df["timestamp"] > prev_t) & (s_df["timestamp"] <= curr_t)
                window_posts = s_df[mask]

                if not window_posts.empty:
                    stats = self.aggregator.aggregate_window(window_posts)
                    agg_scores[i] = stats["sentiment_score"]
                    agg_mentions[i] = stats["mention_count"]
                    agg_dispersion[i] = stats["sentiment_dispersion"]
                    agg_bull_bear[i] = stats.get("bull_bear_diff", stats.get("bull_bear_ratio", 0.0))
                    agg_engagement[i] = stats["total_engagement"]

        # Decay factor: exp(-ln(2) / half_life)
        decay_factor = float(np.exp(-np.log(2.0) / max(1.0, self.half_life_bars)))
        decayed_sentiment = np.zeros(n_bars, dtype=np.float64)
        current_state = 0.0

        for i in range(n_bars):
            if agg_mentions[i] > 0:
                current_state = agg_scores[i]
            else:
                current_state = current_state * decay_factor
            decayed_sentiment[i] = current_state

        # Assign social features (removing collinear momentum feature)
        df[f"{source_prefix}_sentiment_instant"] = agg_scores
        df[f"{source_prefix}_sentiment"] = decayed_sentiment
        df[f"{source_prefix}_mentions"] = agg_mentions
        df[f"{source_prefix}_dispersion"] = agg_dispersion
        df[f"{source_prefix}_bull_bear_diff"] = agg_bull_bear
        df[f"{source_prefix}_bull_bear"] = agg_bull_bear  # alias for backward compatibility
        df[f"{source_prefix}_engagement"] = agg_engagement

        # Exponentially-weighted moving average of decayed sentiment
        df[f"{source_prefix}_sentiment_smooth_3h"] = df[f"{source_prefix}_sentiment"].ewm(span=3, min_periods=1).mean()

        # News Alignment (if provided)
        news_decay_factor = float(np.exp(-np.log(2.0) / max(1.0, self.half_life_bars * 2.0)))
        news_scores = np.zeros(n_bars, dtype=np.float64)
        news_counts = np.zeros(n_bars, dtype=np.int64)
        decayed_news = np.zeros(n_bars, dtype=np.float64)
        news_state = 0.0

        if news_df is not None and not news_df.empty and "timestamp" in news_df.columns:
            n_df = news_df.copy()
            n_df["timestamp"] = pd.to_datetime(n_df["timestamp"], utc=True)
            n_df = n_df.sort_values("timestamp")

            for i in range(n_bars):
                curr_t = bar_timestamps.iloc[i]
                prev_t = bar_timestamps.iloc[i - 1] if i > 0 else (curr_t - pd.Timedelta(hours=72))

                mask = (n_df["timestamp"] > prev_t) & (n_df["timestamp"] <= curr_t)
                window_news = n_df[mask]

                if not window_news.empty and "sentiment_score" in window_news.columns:
                    news_scores[i] = float(window_news["sentiment_score"].mean())
                    news_counts[i] = len(window_news)

        for i in range(n_bars):
            if news_counts[i] > 0:
                news_state = news_scores[i]
            else:
                news_state = news_state * news_decay_factor
            decayed_news[i] = news_state

        df["news_sentiment_instant"] = news_scores
        df["news_sentiment"] = decayed_news
        df["news_count"] = news_counts

        return df
