"""
Unit tests for Multilingual Sentiment Engine and Engagement-Weighted Aggregator.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.sentiment.aggregator import SentimentAggregator
from src.sentiment.scorer import MultilingualSentimentScorer


def test_multilingual_scoring():
    scorer = MultilingualSentimentScorer()

    # English bullish / bearish
    score_en_pos, label_en_pos = scorer.score_text("Great quarterly earnings, strong revenue growth and massive profit surge!")
    assert score_en_pos > 0.3
    assert label_en_pos == "bullish"

    score_en_neg, label_en_neg = scorer.score_text("Severe decline, huge bankruptcy risk and terrible disaster losses")
    assert score_en_neg < -0.3
    assert label_en_neg == "bearish"

    # Traditional Chinese / Cantonese bullish / bearish
    score_zh_pos, label_zh_pos = scorer.score_text("騰訊業績大好，股價爆升破頂創新高，強烈買入！")
    assert score_zh_pos > 0.4
    assert label_zh_pos == "bullish"

    score_zh_neg, label_zh_neg = scorer.score_text("阿里巴巴股價暴跌腰斬，苦主大虧損，沽空割肉！")
    assert score_zh_neg < -0.4
    assert label_zh_neg == "bearish"

    # Korean bullish / bearish
    score_kr_pos, label_kr_pos = scorer.score_text("삼성전자 실적 호조 급등 떡상 가즈아 풀매수!")
    assert score_kr_pos > 0.4
    assert label_kr_pos == "bullish"

    score_kr_neg, label_kr_neg = scorer.score_text("하이닉스 악재 폭락 떡락 패닉셀 손절 망했다")
    assert score_kr_neg < -0.4
    assert label_kr_neg == "bearish"


def test_engagement_weighted_aggregation():
    aggregator = SentimentAggregator()

    # Two posts: Post A is neutral/mildly negative with low engagement,
    # Post B is strongly positive with viral engagement (high likes/views)
    df = pd.DataFrame([
        {
            "sentiment_score": -0.2,
            "likes": 0,
            "replies": 0,
            "views": 10,
            "author_followers": 0,
        },
        {
            "sentiment_score": 0.8,
            "likes": 500,
            "replies": 80,
            "views": 10000,
            "author_followers": 1500,
        },
    ])

    stats = aggregator.aggregate_window(df)

    # Weighted score must be dominated by Post B (> 0.5)
    assert stats["sentiment_score"] > 0.5
    assert stats["mention_count"] == 2
    assert stats["total_engagement"] > 10.0

    # Test with 2 positive posts and 0 negative
    df_pos = pd.DataFrame([
        {"sentiment_score": 0.4, "likes": 10, "replies": 2, "views": 100, "author_followers": 0},
        {"sentiment_score": 0.8, "likes": 500, "replies": 80, "views": 10000, "author_followers": 1500},
    ])
    stats_pos = aggregator.aggregate_window(df_pos)
    assert stats_pos["bull_bear_diff"] > 0.5
