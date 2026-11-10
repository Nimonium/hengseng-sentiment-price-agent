"""
Technical Indicator Engineering and Target Creation.
Target: n+1 next-bar return R_{t+1}.
Features: lagged returns, rolling volatility, volume ratio, high-low spread.
"""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_technical_features(
    price_df: pd.DataFrame,
    lags: list[int] = [1, 2, 3],
    vol_window: int = 5,
) -> pd.DataFrame:
    """
    Compute technical features and next-bar target return on 1h bars.
    """
    df = price_df.copy().sort_index()

    # 1. Bar return R_t = (Close_t - Close_{t-1}) / Close_{t-1}
    df["return_1h"] = df["Close"].pct_change()

    # 2. Target: n+1 next-bar return R_{t+1} = (Close_{t+1} - Close_t) / Close_t
    df["target_return_next"] = df["return_1h"].shift(-1)
    df["target_direction"] = np.where(df["target_return_next"] > 0, 1, 0)
    df["target_price_next"] = df["Close"].shift(-1)

    # 3. Lagged returns
    for lag in lags:
        df[f"return_lag_{lag}"] = df["return_1h"].shift(lag - 1)

    # 4. Rolling volatility
    df["rolling_volatility_5h"] = df["return_1h"].rolling(window=vol_window).std()

    # 5. Volume ratio
    vol_mean = df["Volume"].rolling(window=vol_window).mean()
    df["volume_ratio_5h"] = np.where(vol_mean > 0, df["Volume"] / vol_mean, 1.0)

    # 6. High-Low normalized spread
    df["hl_spread"] = (df["High"] - df["Low"]) / (df["Close"] + 1e-8)

    # 7. Close vs Open return within bar
    df["bar_intraday_return"] = (df["Close"] - df["Open"]) / (df["Open"] + 1e-8)

    return df

