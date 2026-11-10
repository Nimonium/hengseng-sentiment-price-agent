"""
Unit tests for Support/Resistance calculation and Divergence Detection.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.features.levels import SupportResistanceCalculator


def test_pivot_points_calculation():
    # Synthetic dataframe with known High=110, Low=90, Close=100
    df = pd.DataFrame({
        "Open": [95.0, 98.0, 99.0],
        "High": [105.0, 110.0, 108.0],
        "Low": [92.0, 90.0, 94.0],
        "Close": [98.0, 100.0, 100.0],
        "Volume": [1000, 2000, 1500],
    })

    levels = SupportResistanceCalculator.compute_levels(df, swing_window=3)

    # High=110, Low=90, Close=100
    # Pivot = (110 + 90 + 100) / 3 = 100.0
    # R1 = 2*100 - 90 = 110.0
    # S1 = 2*100 - 110 = 90.0
    # R2 = 100 + (110 - 90) = 120.0
    # S2 = 100 - (110 - 90) = 80.0
    assert levels["pivot_point"] == 100.0
    assert levels["resistance_1"] == 110.0
    assert levels["support_1"] == 90.0
    assert levels["resistance_2"] == 120.0
    assert levels["support_2"] == 80.0
    assert levels["swing_high"] == 110.0
    assert levels["swing_low"] == 90.0


def test_divergence_detector():
    # Bullish divergence: price dropped (-1.5%) while sentiment is strongly positive (+0.4)
    has_div, note = SupportResistanceCalculator.detect_divergence(-0.015, 0.40)
    assert has_div is True
    assert "Bullish Divergence" in note

    # Bearish divergence: price rallied (+2.0%) while sentiment is negative (-0.35)
    has_div_bear, note_bear = SupportResistanceCalculator.detect_divergence(0.020, -0.35)
    assert has_div_bear is True
    assert "Bearish Divergence" in note_bear

    # Normal state
    has_norm, _ = SupportResistanceCalculator.detect_divergence(0.005, 0.05)
    assert has_norm is False

