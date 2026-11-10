"""
Engagement-Weighted Sentiment Aggregation per Ticker per Time Bar.
Computes weighted sentiment score, volume, dispersion, and normalized bull-bear differential.
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SentimentAggregator:
    def __init__(self, weights_config: Dict[str, float] = None):
        cfg = weights_config or {}
        self.w_base = cfg.get("base", 1.0)
        self.w_likes = cfg.get("likes", 1.0)
        self.w_replies = cfg.get("replies", 1.5)
        self.w_views = cfg.get("views", 0.5)
        self.w_followers = cfg.get("followers", 0.3)

    def compute_engagement_weight(self, row: pd.Series) -> float:
        """
        Compute non-linear engagement weight:
        w = base + w_likes*ln(1+likes) + w_replies*ln(1+replies) + w_views*ln(1+views)
        """
        likes = float(row.get("likes", 0) or 0)
        replies = float(row.get("replies", 0) or 0)
        views = float(row.get("views", 0) or 0)
        followers = float(row.get("author_followers", 0) or 0)

        w = self.w_base
        if likes > 0:
            w += self.w_likes * np.log1p(likes)
        if replies > 0:
            w += self.w_replies * np.log1p(replies)
        if views > 0:
            w += self.w_views * np.log1p(views)
        if followers > 0:
            w += self.w_followers * np.log1p(followers)

        return float(w)

    def aggregate_window(self, posts_df: pd.DataFrame) -> Dict[str, float]:
        """
        Aggregate a set of posts that occurred in a specific time window.
        Returns dictionary of computed features.
        """
        if posts_df.empty or "sentiment_score" not in posts_df.columns:
            return {
                "sentiment_score": 0.0,
                "mention_count": 0,
                "sentiment_dispersion": 0.0,
                "bull_bear_diff": 0.0,
                "total_engagement": 0.0,
            }

        scores = posts_df["sentiment_score"].to_numpy()
        weights = np.array([self.compute_engagement_weight(r) for _, r in posts_df.iterrows()])

        total_weight = np.sum(weights)
        if total_weight > 0:
            weighted_score = float(np.sum(scores * weights) / total_weight)
        else:
            weighted_score = float(np.mean(scores))

        mention_count = len(posts_df)
        dispersion = float(np.std(scores)) if mention_count > 1 else 0.0

        n_pos = float(np.sum(scores > 0.05))
        n_neg = float(np.sum(scores < -0.05))
        # Bounded normalized differential: (bull - bear) / (bull + bear + 1.0) in [-1, 1]
        bull_bear_diff = float((n_pos - n_neg) / (n_pos + n_neg + 1.0))

        return {
            "sentiment_score": round(weighted_score, 4),
            "mention_count": mention_count,
            "sentiment_dispersion": round(dispersion, 4),
            "bull_bear_diff": round(bull_bear_diff, 4),
            "total_engagement": round(total_weight, 2),
        }
