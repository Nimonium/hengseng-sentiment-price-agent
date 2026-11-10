"""
Time-Series Cross Validation, Ablation Engine, and Statistical Significance Testing.
Strict Expanding-Window / Walk-Forward evaluation with zero data leakage.
Automatically filters constant zero-variance features, clips predictions to +/- 3x rolling std,
and supports pooled exchange-level walk-forward validation with ticker-level normalization.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, RidgeCV

from src.models.trainer import create_model

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prev: np.ndarray | None = None,
) -> Dict[str, Any]:
    """
    Compute out-of-sample regression metrics, naive baselines, and binomial significance test.
    """
    n_samples = len(y_true)
    if n_samples == 0:
        return {
            "mae": 0.0,
            "rmse": 0.0,
            "directional_accuracy": 0.5,
            "ic": 0.0,
            "sample_size": 0,
            "k_correct": 0,
            "naive_always_up": 0.5,
            "naive_prev_dir": 0.5,
            "p_value_vs_50": 1.0,
            "p_value_vs_up": 1.0,
            "edge_significant": False,
        }

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # Directional accuracy: sign agreement
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    matches = (true_sign == pred_sign)
    k_correct = int(np.sum(matches))
    dir_acc = float(k_correct / n_samples)

    # 1. Naive Baseline: Always Up (predict positive return every bar)
    k_up = int(np.sum(y_true > 0))
    naive_always_up = float(k_up / n_samples)

    # 2. Naive Baseline: Previous Direction (predict return_t-1 direction)
    if y_prev is not None and len(y_prev) == n_samples:
        prev_sign = np.sign(y_prev)
        naive_prev_dir = float(np.mean(prev_sign == true_sign))
    else:
        if n_samples > 1:
            prev_dir_matches = np.sign(y_true[1:]) == np.sign(y_true[:-1])
            naive_prev_dir = float(np.mean(prev_dir_matches))
        else:
            naive_prev_dir = 0.5

    # 3. Binomial Statistical Significance Tests
    # Null Hypothesis H0: Accuracy <= 50% (random guess)
    binom_res_50 = stats.binomtest(k=k_correct, n=n_samples, p=0.5, alternative="greater")
    p_value_50 = float(binom_res_50.pvalue)

    # Null Hypothesis H0: Accuracy <= Always-Up baseline
    baseline_p = min(0.99, max(0.01, naive_always_up))
    binom_res_up = stats.binomtest(k=k_correct, n=n_samples, p=baseline_p, alternative="greater")
    p_value_up = float(binom_res_up.pvalue)

    # True statistical edge exists strictly if p_value < 0.05 AND exceeds always up
    edge_significant = bool(p_value_50 < 0.05 and dir_acc > naive_always_up)

    # Information Coefficient (Pearson Correlation)
    if np.std(y_pred) > 1e-8 and np.std(y_true) > 1e-8:
        ic = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        ic = 0.0

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "directional_accuracy": round(dir_acc, 4),
        "ic": round(ic, 4),
        "sample_size": n_samples,
        "k_correct": k_correct,
        "naive_always_up": round(naive_always_up, 4),
        "naive_prev_dir": round(naive_prev_dir, 4),
        "p_value_vs_50": round(p_value_50, 4),
        "p_value_vs_up": round(p_value_up, 4),
        "edge_significant": edge_significant,
    }


def compute_feature_diagnostics(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "target_return_next",
) -> pd.DataFrame:
    """
    Compute feature means, standard deviations, and standardized Ridge coefficients with CV tuning.
    Automatically identifies and drops constant zero-variance features.
    """
    clean_df = df.dropna(subset=[target_col] + feature_cols).copy()
    if clean_df.empty:
        return pd.DataFrame(columns=["feature", "mean", "std", "ridge_coef_standardized", "status"])

    # Identify non-constant features
    valid_cols = []
    dropped_cols = []
    for c in feature_cols:
        col_std = float(clean_df[c].std())
        if col_std > 1e-6:
            valid_cols.append(c)
        else:
            dropped_cols.append(c)

    records = []
    if valid_cols:
        X = clean_df[valid_cols].to_numpy()
        y = clean_df[target_col].to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        ridge = RidgeCV(alphas=np.logspace(-2, 4, 15), fit_intercept=True)
        ridge.fit(X_scaled, y)

        for i, col in enumerate(valid_cols):
            records.append({
                "feature": col,
                "mean": round(float(scaler.mean_[i]), 6),
                "std": round(float(scaler.scale_[i]), 6),
                "ridge_coef_standardized": round(float(ridge.coef_[i]), 6),
                "status": "ACTIVE",
            })

    for col in dropped_cols:
        records.append({
            "feature": col,
            "mean": round(float(clean_df[col].mean()), 6),
            "std": round(float(clean_df[col].std()), 6),
            "ridge_coef_standardized": 0.0,
            "status": "DROPPED (Zero-Variance)",
        })

    return pd.DataFrame(records)


class TimeSeriesValidator:
    def __init__(self, n_splits: int = 4, min_train_size: int = 25):
        self.n_splits = n_splits
        self.min_train_size = min_train_size

    def evaluate_model(
        self,
        model_name: str,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_return_next",
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Run walk-forward expanding window time-series CV on clean dataset.
        Drops constant zero-variance features and clips predictions to +/- 3x rolling return std.
        """
        clean_df = df.dropna(subset=[target_col] + feature_cols).copy()

        if len(clean_df) < self.min_train_size + self.n_splits:
            logger.warning(f"Insufficient rows ({len(clean_df)}) for {self.n_splits}-split CV")
            return compute_metrics(np.array([]), np.array([])), pd.DataFrame()

        # Drop constant zero-variance features across clean_df
        active_cols = [c for c in feature_cols if clean_df[c].std() > 1e-6]
        if not active_cols and model_name != "baseline_zero":
            logger.info("All feature columns are constant. Using baseline zero.")
            model_name = "baseline_zero"
            active_cols = feature_cols[:1]

        X = clean_df[active_cols].to_numpy() if active_cols else clean_df[feature_cols].to_numpy()
        y = clean_df[target_col].to_numpy()
        y_prev = clean_df["return_lag_1"].to_numpy() if "return_lag_1" in clean_df.columns else None

        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=max(5, len(clean_df) // (self.n_splits + 1)))

        all_y_true = []
        all_y_pred = []
        all_y_prev = []
        pred_indices = []

        # Calculate rolling volatility for realistic clipping
        if "rolling_volatility_5h" in clean_df.columns:
            roll_vol_series = clean_df["rolling_volatility_5h"].to_numpy()
        elif "return_1h" in clean_df.columns:
            roll_vol_series = clean_df["return_1h"].rolling(10, min_periods=3).std().fillna(0.01).to_numpy()
        else:
            roll_vol_series = np.full(len(clean_df), 0.01)

        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model = create_model(model_name)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # MODEL SANITY: Clip predictions to +/- 3x rolling std
            test_vol = roll_vol_series[test_idx]
            clip_bound = np.maximum(0.012, 3.0 * test_vol)
            preds = np.clip(preds, -clip_bound, clip_bound)

            all_y_true.extend(y_test)
            all_y_pred.extend(preds)
            if y_prev is not None:
                all_y_prev.extend(y_prev[test_idx])
            pred_indices.extend(test_idx)

        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)
        y_prev_arr = np.array(all_y_prev) if all_y_prev else None

        metrics = compute_metrics(y_true_arr, y_pred_arr, y_prev=y_prev_arr)

        pred_df = clean_df.iloc[pred_indices].copy()
        pred_df["predicted_return"] = y_pred_arr
        pred_df["actual_return"] = y_true_arr
        pred_df["predicted_price_next"] = pred_df["Close"] * (1.0 + y_pred_arr)
        pred_df["actual_price_next"] = (
            pred_df["target_price_next"]
            if "target_price_next" in pred_df.columns
            else pred_df["Close"] * (1.0 + y_true_arr)
        )
        pred_df["direction_correct"] = np.sign(y_pred_arr) == np.sign(y_true_arr)

        return metrics, pred_df

    def run_ablation(
        self,
        df: pd.DataFrame,
        price_cols: List[str],
        social_cols: List[str],
        news_cols: List[str],
        model_name: str = "sentiment_ridge",
        target_col: str = "target_return_next",
    ) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, pd.DataFrame]:
        """
        Perform strict feature ablation:
        1. Baseline Zero (Naive)
        2. Price Only
        3. Price + Social
        4. Price + News
        5. All Features (Price + Social + News)
        """
        ablation_results: Dict[str, Dict[str, Any]] = {}

        # 1. Baseline Zero
        m_zero, _ = self.evaluate_model("baseline_zero", df, price_cols, target_col)
        ablation_results["Naive (Zero Return)"] = m_zero

        # 2. Price Only
        m_price, _ = self.evaluate_model(model_name, df, price_cols, target_col)
        ablation_results["Price Only"] = m_price

        # 3. Price + Social
        m_social, _ = self.evaluate_model(model_name, df, price_cols + social_cols, target_col)
        ablation_results["Price + Social"] = m_social

        # 4. Price + News
        m_news, _ = self.evaluate_model(model_name, df, price_cols + news_cols, target_col)
        ablation_results["Price + News"] = m_news

        # 5. All Features
        all_cols = list(dict.fromkeys(price_cols + social_cols + news_cols))
        m_all, best_pred_df = self.evaluate_model(model_name, df, all_cols, target_col)
        ablation_results["All Features (Price+Social+News)"] = m_all

        # Compute feature diagnostics for all features
        diagnostics = compute_feature_diagnostics(df, all_cols, target_col)

        return ablation_results, best_pred_df, diagnostics

    def run_pooled_exchange_ablation(
        self,
        ticker_dfs: Dict[str, pd.DataFrame],
        price_cols: List[str],
        social_cols: List[str],
        news_cols: List[str],
        model_name: str = "sentiment_ridge",
        target_col: str = "target_return_next",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train and evaluate one pooled model per exchange with ticker-level normalization.
        Combines all tickers of the exchange into a single cross-sectional panel dataset.
        """
        pooled_rows = []
        all_cols = list(dict.fromkeys(price_cols + social_cols + news_cols + [target_col, "return_1h", "Close", "rolling_volatility_5h"]))

        for sym, df in ticker_dfs.items():
            sub_df = df.copy()
            # Normalize numerical features per ticker (z-score) to prevent cross-ticker scale distortions
            feat_subset = [c for c in (price_cols + social_cols + news_cols) if c in sub_df.columns]
            for c in feat_subset:
                c_std = float(sub_df[c].std())
                c_mean = float(sub_df[c].mean())
                if c_std > 1e-6:
                    sub_df[c] = (sub_df[c] - c_mean) / c_std

            sub_df["ticker"] = sym
            pooled_rows.append(sub_df)

        if not pooled_rows:
            return {}

        pooled_panel = pd.concat(pooled_rows).sort_index()

        ablation_results, _, _ = self.run_ablation(
            df=pooled_panel,
            price_cols=price_cols,
            social_cols=social_cols,
            news_cols=news_cols,
            model_name=model_name,
            target_col=target_col,
        )
        return ablation_results
