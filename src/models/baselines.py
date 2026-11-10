"""
Baseline Models for Benchmark Comparison.
Includes Zero-Return Naive Baseline and Price-Only Linear Baseline.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge


class ZeroReturnBaseline(BaseEstimator, RegressorMixin):
    """
    Naive Baseline: Predicts zero return (persistence: next price equals current price).
    Standard benchmark in intraday equity returns where mean return is ~ 0.
    """
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=np.float64)


class PriceOnlyBaseline(BaseEstimator, RegressorMixin):
    """
    Price-Only Baseline: Ridge regression using strictly technical/price features.
    No sentiment or news signals.
    """
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model = Ridge(alpha=self.alpha)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

