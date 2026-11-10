"""
Unit tests for TimeSeriesValidator, compute_metrics, and Binomial statistical testing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.validator import compute_metrics, compute_feature_diagnostics, TimeSeriesValidator


def test_compute_metrics_binomial_significance():
    # Synthetic ground truth and predictions with 50% accuracy
    y_true = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.01])
    y_pred = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # Always up

    metrics = compute_metrics(y_true, y_pred)
    assert metrics["sample_size"] == 8
    assert metrics["k_correct"] == 4
    assert metrics["directional_accuracy"] == 0.5
    assert metrics["naive_always_up"] == 0.5
    assert metrics["p_value_vs_50"] >= 0.5
    assert metrics["edge_significant"] is False


def test_feature_diagnostics_calculation():
    # DataFrame with synthetic features
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "target_return_next": np.random.normal(0, 0.01, n),
        "feat_price": np.random.normal(0, 0.02, n),
        "feat_social": np.random.normal(0, 0.05, n),
        "feat_constant": np.zeros(n),
    })

    diagnostics = compute_feature_diagnostics(
        df=df,
        feature_cols=["feat_price", "feat_social", "feat_constant"],
        target_col="target_return_next",
    )

    assert len(diagnostics) == 3
    assert "ridge_coef_standardized" in diagnostics.columns
    # Check that constant feature is identified and dropped
    const_row = diagnostics[diagnostics["feature"] == "feat_constant"].iloc[0]
    assert const_row["std"] == 0.0
    assert const_row["status"] == "DROPPED (Zero-Variance)"
    assert const_row["ridge_coef_standardized"] == 0.0
