"""
Unit tests for zero look-ahead leakage and point-in-time causal alignment.
"""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np
import pytest

from src.features.aligner import FeatureAligner
from src.sentiment.aggregator import SentimentAggregator


def create_dummy_price_df(n_bars: int = 10, start_dt: str = "2026-11-01 09:30:00") -> pd.DataFrame:
    """Create a synthetic 1h price bar series."""
    base_time = pd.to_datetime(start_dt, utc=True)
    index = [base_time + pd.Timedelta(hours=i) for i in range(n_bars)]
    
    data = {
        "Open": [100.0 + i for i in range(n_bars)],
        "High": [102.0 + i for i in range(n_bars)],
        "Low": [99.0 + i for i in range(n_bars)],
        "Close": [101.0 + i for i in range(n_bars)],
        "Volume": [1000 * (i + 1) for i in range(n_bars)],
    }
    df = pd.DataFrame(data, index=index)
    df.index.name = "timestamp_utc"
    return df


def test_zero_lookahead_future_posts_do_not_leak():
    """
    CRITICAL TEST: Ensure that adding future posts at time T_future does NOT
    change any sentiment features for bars occurring before T_future.
    """
    price_df = create_dummy_price_df(n_bars=6)
    aligner = FeatureAligner()

    # Base posts occurring in bars 1 and 2
    t0 = price_df.index[0]
    base_posts = pd.DataFrame([
        {
            "id": "post_1",
            "timestamp": t0 + timedelta(minutes=15),
            "sentiment_score": 0.5,
            "likes": 10,
            "replies": 2,
            "views": 100,
            "author_followers": 0,
        },
        {
            "id": "post_2",
            "timestamp": t0 + timedelta(minutes=45),
            "sentiment_score": -0.2,
            "likes": 5,
            "replies": 1,
            "views": 50,
            "author_followers": 0,
        },
    ])

    # Run alignment with base posts
    df_base = aligner.align_sentiment_to_bars(price_df, base_posts)

    # Now create a future extreme post at bar 5 (T0 + 5 hours)
    future_post = pd.DataFrame([
        {
            "id": "future_extreme_leak_attempt",
            "timestamp": t0 + timedelta(hours=4, minutes=30),  # In bar 4
            "sentiment_score": 1.0,  # Extreme bullish
            "likes": 100000,
            "replies": 5000,
            "views": 1000000,
            "author_followers": 50000,
        }
    ])
    augmented_posts = pd.concat([base_posts, future_post], ignore_index=True)

    # Run alignment with future post added
    df_augmented = aligner.align_sentiment_to_bars(price_df, augmented_posts)

    # Assert that bars 0, 1, 2, 3, 4 have IDENTICAL features
    # Bar 5 may change, but earlier bars MUST be strictly invariant
    for col in ["social_sentiment", "social_mentions", "social_dispersion", "social_bull_bear"]:
        np.testing.assert_array_almost_equal(
            df_base[col].iloc[:5].to_numpy(),
            df_augmented[col].iloc[:5].to_numpy(),
            err_msg=f"Lookahead leakage detected in column {col} for early bars!"
        )

    # Bar 5 (covering (t0+4h, t0+5h]) must capture the future post
    assert df_augmented["social_sentiment"].iloc[5] > 0.95
    assert df_augmented["social_mentions"].iloc[5] == 1


def test_empty_posts_handling():
    """Verify that empty social or news posts return clean zeroes without crashing."""
    price_df = create_dummy_price_df(n_bars=4)
    aligner = FeatureAligner()

    empty_df = pd.DataFrame(columns=["id", "timestamp", "sentiment_score", "likes", "replies", "views"])
    aligned = aligner.align_sentiment_to_bars(price_df, empty_df)

    assert (aligned["social_sentiment"] == 0.0).all()
    assert (aligned["social_mentions"] == 0).all()
    assert (aligned["news_sentiment"] == 0.0).all()


def test_exact_boundary_conditions():
    """Verify posts occurring exactly at bar boundary timestamp."""
    price_df = create_dummy_price_df(n_bars=3)
    aligner = FeatureAligner()

    t1 = price_df.index[1]
    boundary_post = pd.DataFrame([
        {
            "id": "post_exact_boundary",
            "timestamp": t1,  # Exactly at bar 1 timestamp
            "sentiment_score": 0.8,
            "likes": 20,
            "replies": 5,
            "views": 200,
            "author_followers": 0,
        }
    ])

    aligned = aligner.align_sentiment_to_bars(price_df, boundary_post)

    # Bar 0 should be 0, Bar 1 should include the boundary post
    assert aligned["social_sentiment"].iloc[0] == 0.0
    assert aligned["social_mentions"].iloc[0] == 0
    assert aligned["social_mentions"].iloc[1] == 1
    assert aligned["social_sentiment"].iloc[1] > 0.7
