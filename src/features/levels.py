"""
Support and Resistance Level Calculations using Standard Pivot Points and Rolling Swings.
Also computes creative signals: Sentiment vs Price Divergence and Model Confidence.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SupportResistanceCalculator:
    """
    Computes technical pivot points (Floor Trader standard pivots)
    and rolling swing extrema (local highs/lows) with human-readable methodology.
    """

    @staticmethod
    def compute_levels(price_df: pd.DataFrame, swing_window: int = 10) -> Dict[str, Any]:
        """
        Calculate support and resistance levels from the recent price series.
        """
        if price_df.empty or len(price_df) < 1:
            return {}

        df = price_df.copy().sort_index()
        recent = df.iloc[-min(len(df), 24):]  # Last 24 hours of bars

        high = float(recent["High"].max())
        low = float(recent["Low"].min())
        close = float(recent["Close"].iloc[-1])

        # 1. Floor Trader Pivot Points
        # P = (H + L + C) / 3
        # R1 = 2P - L, S1 = 2P - H
        # R2 = P + (H - L), S2 = P - (H - L)
        # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
        pivot = (high + low + close) / 3.0
        r1 = 2.0 * pivot - low
        s1 = 2.0 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        r3 = high + 2.0 * (pivot - low)
        s3 = low - 2.0 * (high - pivot)

        # 2. Rolling Swing Highs & Lows
        swing_high = float(df["High"].rolling(window=swing_window, min_periods=3).max().iloc[-1])
        swing_low = float(df["Low"].rolling(window=swing_window, min_periods=3).min().iloc[-1])

        return {
            "current_price": round(close, 2),
            "pivot_point": round(pivot, 2),
            "resistance_1": round(r1, 2),
            "resistance_2": round(r2, 2),
            "resistance_3": round(r3, 2),
            "support_1": round(s1, 2),
            "support_2": round(s2, 2),
            "support_3": round(s3, 2),
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "methodology": (
                "Support & Resistance derived from classical 24h Floor Trader Pivot Points "
                "(P = [H+L+C]/3, R1=2P-L, S1=2P-H, R2=P+[H-L], S2=P-[H-L]) blended with rolling "
                f"{swing_window}-bar price action extrema (swing high/low channels)."
            )
        }

    @staticmethod
    def detect_divergence(
        recent_price_return: float,
        recent_sentiment: float,
        threshold_price: float = 0.008,
        threshold_sentiment: float = 0.25
    ) -> Tuple[bool, str]:
        """
        Detect Sentiment vs Price Divergence:
        - Bullish Divergence: Price dropped while sentiment is strongly positive (potential rebound).
        - Bearish Divergence: Price rallied while sentiment is strongly negative (potential exhaustion).
        """
        if recent_price_return < -threshold_price and recent_sentiment > threshold_sentiment:
            return True, "Bullish Divergence: Price fell while sentiment surged positive (buy dip potential)"
        elif recent_price_return > threshold_price and recent_sentiment < -threshold_sentiment:
            return True, "Bearish Divergence: Price climbed while sentiment deteriorated (exhaustion alert)"
        return False, "Normal alignment between price action and social sentiment"
