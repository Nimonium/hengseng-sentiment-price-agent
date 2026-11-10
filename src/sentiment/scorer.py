"""
Multilingual Sentiment Scoring Engine.
Supports English, Traditional Chinese / Cantonese, and Korean.
Outputs normalized continuous score in [-1.0, 1.0] and categorical label.
Includes caching to avoid re-scoring duplicate texts.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Specialized Financial & Social Lexicons
CHINESE_FINANCIAL_LEXICON = {
    # Bullish (Cantonese & Traditional/Simplified Chinese)
    "升": 0.6, "爆升": 0.9, "破頂": 0.85, "牛": 0.7, "看好": 0.7, "買入": 0.8, "追入": 0.75,
    "反彈": 0.6, "大賺": 0.8, "利好": 0.75, "業績好": 0.8, "暴賺": 0.9, "飛天": 0.85,
    "好消息": 0.65, "增長": 0.5, "回升": 0.55, "入市": 0.6, "加倉": 0.7, "上漲": 0.6,
    "創新高": 0.85, "大陽線": 0.8, "多頭": 0.7, "支持": 0.4, "低估": 0.5,
    # Bearish
    "跌": -0.6, "暴跌": -0.9, "大跌": -0.85, "熊": -0.7, "看淡": -0.7, "賣出": -0.8, "沽空": -0.85,
    "跳水": -0.8, "苦主": -0.75, "割肉": -0.8, "止蝕": -0.6, "腰斬": -0.9, "爆煲": -0.9,
    "利空": -0.75, "業績差": -0.8, "虧損": -0.7, "垃圾": -0.75, "破底": -0.85, "陰燭": -0.6,
    "空頭": -0.7, "套牢": -0.8, "崩盤": -0.95, "下跌": -0.6, "被坑": -0.7,
}

KOREAN_FINANCIAL_LEXICON = {
    # Bullish Korean
    "상승": 0.65, "급등": 0.9, "폭등": 0.95, "호재": 0.8, "매수": 0.75, "떡상": 0.9,
    "반등": 0.6, "수익": 0.7, "대박": 0.85, "실적호조": 0.8, "돌파": 0.7, "신고가": 0.85,
    "풀매수": 0.85, "추매": 0.6, "존버": 0.5, "불장": 0.8, "외인매수": 0.7, "기관매수": 0.7,
    "목표가상향": 0.8, "가즈아": 0.8, "양봉": 0.7, "우상향": 0.75,
    # Bearish Korean
    "하락": -0.65, "급락": -0.9, "폭락": -0.95, "악재": -0.8, "매도": -0.75, "떡락": -0.9,
    "손절": -0.7, "손실": -0.7, "물림": -0.75, "개잡주": -0.85, "실적쇼크": -0.85, "신저가": -0.85,
    "패닉셀": -0.9, "투매": -0.85, "탈출": -0.6, "하한가": -0.95, "음봉": -0.7, "우하향": -0.75,
    "개미털기": -0.6, "적자": -0.7, "망함": -0.85, "외인매도": -0.7,
}


class MultilingualSentimentScorer:
    def __init__(self, gemini_api_key: Optional[str] = None):
        self.vader = SentimentIntensityAnalyzer()
        self.gemini_api_key = gemini_api_key
        self._cache: Dict[str, Tuple[float, str]] = {}

    def score_text(self, text: str) -> Tuple[float, str]:
        """
        Calculate multilingual sentiment score in [-1.0, 1.0] and category (bullish/bearish/neutral).
        """
        if not text or not text.strip():
            return 0.0, "neutral"

        text_clean = text.strip()
        text_hash = hashlib.md5(text_clean.encode("utf-8")).hexdigest()
        if text_hash in self._cache:
            return self._cache[text_hash]

        # 1. Check CJK / Korean presence
        has_korean = bool(re.search(r"[\uac00-\ud7a3]", text_clean))
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", text_clean))

        lexicon_score = 0.0
        match_count = 0

        if has_korean:
            for word, weight in KOREAN_FINANCIAL_LEXICON.items():
                if word in text_clean:
                    lexicon_score += weight
                    match_count += 1

        if has_chinese:
            for word, weight in CHINESE_FINANCIAL_LEXICON.items():
                if word in text_clean:
                    lexicon_score += weight
                    match_count += 1

        # 2. English / VADER component
        vader_res = self.vader.polarity_scores(text_clean)
        vader_compound = vader_res["compound"]

        # 3. Blend scores
        if match_count > 0:
            avg_lex = lexicon_score / match_count
            if abs(vader_compound) > 0.05:
                final_score = 0.65 * avg_lex + 0.35 * vader_compound
            else:
                final_score = avg_lex
        else:
            final_score = vader_compound

        # Clip to [-1.0, 1.0]
        final_score = max(-1.0, min(1.0, round(final_score, 4)))

        # Assign label
        if final_score >= 0.15:
            label = "bullish"
        elif final_score <= -0.15:
            label = "bearish"
        else:
            label = "neutral"

        self._cache[text_hash] = (final_score, label)
        return final_score, label

    def score_dataframe(self, df, text_column: str = "text"):
        """Score an entire dataframe and append sentiment_score and sentiment_label columns."""
        if df.empty or text_column not in df.columns:
            df["sentiment_score"] = 0.0
            df["sentiment_label"] = "neutral"
            return df

        scores = []
        labels = []
        for text in df[text_column]:
            s, l = self.score_text(str(text))
            scores.append(s)
            labels.append(l)

        df["sentiment_score"] = scores
        df["sentiment_label"] = labels
        return df

