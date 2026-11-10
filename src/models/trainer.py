"""
Model Factory and Training Utilities for Intraday Price Prediction.
Implements RidgeCV with training-fold-only regularization tuning and tree baselines.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb

from src.models.baselines import ZeroReturnBaseline, PriceOnlyBaseline

logger = logging.getLogger(__name__)


def create_model(model_name: str, random_state: int = 42) -> Any:
    """Instantiate a standardized model pipeline by name."""
    if model_name == "baseline_zero":
        return ZeroReturnBaseline()
    elif model_name == "baseline_price_ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 15))),
        ])
    elif model_name == "sentiment_ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-2, 4, 15))),
        ])
    elif model_name == "sentiment_lgbm":
        return lgb.LGBMRegressor(
            n_estimators=35,
            max_depth=3,
            learning_rate=0.05,
            min_child_samples=5,
            random_state=random_state,
            verbosity=-1,
        )
    elif model_name == "sentiment_xgboost":
        return xgb.XGBRegressor(
            n_estimators=35,
            max_depth=3,
            learning_rate=0.05,
            random_state=random_state,
            verbosity=0,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
