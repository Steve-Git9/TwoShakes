"""
Feature Engineering Transformations
=====================================
All sklearn-backed transformations that operate on pandas DataFrames.
Every public function returns (updated_df, FeatureEngineeringRecord).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.schemas import FeatureEngineeringAction, FeatureEngineeringRecord, FeatureEngineeringActionType

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _record(
    action_id: str,
    action_type: FeatureEngineeringActionType,
    columns_affected: list[str],
    columns_added: list[str],
    columns_removed: list[str],
    rows_before: int,
    rows_after: int,
    cols_before: int,
    cols_after: int,
    success: bool,
    error_message: str | None = None,
) -> FeatureEngineeringRecord:
    return FeatureEngineeringRecord(
        action_id=action_id,
        action_type=action_type,
        columns_affected=columns_affected,
        columns_added=columns_added,
        columns_removed=columns_removed,
        rows_before=rows_before,
        rows_after=rows_after,
        cols_before=cols_before,
        cols_after=cols_after,
        success=success,
        error_message=error_message,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────

def one_hot_encode(
    df: pd.DataFrame, column: str, action_id: str = "fe_ohe",
    drop_first: bool = True,
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """One-hot encode a categorical column."""
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    dummies = pd.get_dummies(df[column], prefix=column, drop_first=drop_first, dtype=int)
    df = df.drop(columns=[column]).join(dummies)
    added = list(dummies.columns)
    return df, _record(
        action_id, FeatureEngineeringActionType.ONE_HOT_ENCODE,
        [column], added, [column],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


def label_encode(
    df: pd.DataFrame, column: str, action_id: str = "fe_le",
    order: list | None = None,
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Label encode an ordinal column. order= provides explicit mapping."""
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    if order:
        mapping = {v: i for i, v in enumerate(order)}
        df[column] = df[column].map(mapping).fillna(-1).astype(int)
    else:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
    return df, _record(
        action_id, FeatureEngineeringActionType.LABEL_ENCODE,
        [column], [], [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


def ordinal_encode(
    df: pd.DataFrame, column: str, action_id: str = "fe_oe",
    mapping: dict | None = None,
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Ordinal encode with explicit mapping dict."""
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    if mapping:
        df[column] = df[column].map(mapping).fillna(-1).astype(int)
    else:
        from sklearn.preprocessing import OrdinalEncoder
        enc = OrdinalEncoder()
        df[[column]] = enc.fit_transform(df[[column]].astype(str))
    return df, _record(
        action_id, FeatureEngineeringActionType.ORDINAL_ENCODE,
        [column], [], [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


def target_encode(
    df: pd.DataFrame, column: str, action_id: str = "fe_te",
    target_column: str = "", smoothing: float = 1.0,
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Replace categories with smoothed mean of target variable."""
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    if not target_column or target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in DataFrame")
    global_mean = df[target_column].mean()
    stats = df.groupby(column)[target_column].agg(["mean", "count"])
    smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
    df[column] = df[column].map(smoothed).fillna(global_mean)
    return df, _record(
        action_id, FeatureEngineeringActionType.TARGET_ENCODE,
        [column], [], [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


def frequency_encode(
    df: pd.DataFrame, column: str, action_id: str = "fe_freq",
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Replace categories with their frequency counts."""
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    freq = df[column].value_counts()
    df[column] = df[column].map(freq)
    return df, _record(
        action_id, FeatureEngineeringActionType.FREQUENCY_ENCODE,
        [column], [], [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Scaling
# ─────────────────────────────────────────────────────────────────────────────

def _apply_scaler(df, column, scaler, action_type, action_id):
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    vals = pd.to_numeric(df[column], errors="coerce").fillna(0).values.reshape(-1, 1)
    df[column] = scaler.fit_transform(vals).ravel()
    return df, _record(action_id, action_type, [column], [], [], rows_b, len(df), cols_b, len(df.columns), True)


def min_max_scale(df, column, action_id="fe_mms", feature_range=(0, 1)):
    from sklearn.preprocessing import MinMaxScaler
    return _apply_scaler(df, column, MinMaxScaler(feature_range=tuple(feature_range)),
                         FeatureEngineeringActionType.MIN_MAX_SCALE, action_id)


def standard_scale(df, column, action_id="fe_ss"):
    from sklearn.preprocessing import StandardScaler
    return _apply_scaler(df, column, StandardScaler(),
                         FeatureEngineeringActionType.STANDARD_SCALE, action_id)


def robust_scale(df, column, action_id="fe_rs"):
    from sklearn.preprocessing import RobustScaler
    return _apply_scaler(df, column, RobustScaler(),
                         FeatureEngineeringActionType.ROBUST_SCALE, action_id)


def max_abs_scale(df, column, action_id="fe_mas"):
    from sklearn.preprocessing import MaxAbsScaler
    return _apply_scaler(df, column, MaxAbsScaler(),
                         FeatureEngineeringActionType.MAX_ABS_SCALE, action_id)


# ─────────────────────────────────────────────────────────────────────────────
# Distribution transforms
# ─────────────────────────────────────────────────────────────────────────────

def log_transform(
    df: pd.DataFrame, column: str, action_id: str = "fe_log",
    base: str = "natural",
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Log transform for right-skewed data. Auto-adds +1 offset if zeros present."""
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    vals = pd.to_numeric(df[column], errors="coerce").fillna(0)
    offset = 0.0
    if (vals <= 0).any():
        offset = abs(vals.min()) + 1.0
        logger.info(f"[log_transform] Adding offset {offset:.4f} to {column}")
    if base == "natural":
        df[column] = np.log(vals + offset)
    else:
        df[column] = np.log10(vals + offset)
    return df, _record(
        action_id, FeatureEngineeringActionType.LOG_TRANSFORM,
        [column], [], [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


def power_transform(
    df: pd.DataFrame, column: str, action_id: str = "fe_pt",
    method: str = "yeo-johnson",
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Yeo-Johnson or Box-Cox power transform."""
    from sklearn.preprocessing import PowerTransformer
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    vals = pd.to_numeric(df[column], errors="coerce").fillna(0).values.reshape(-1, 1)
    pt = PowerTransformer(method=method)
    df[column] = pt.fit_transform(vals).ravel()
    return df, _record(
        action_id, FeatureEngineeringActionType.POWER_TRANSFORM,
        [column], [], [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


def quantile_transform(
    df: pd.DataFrame, column: str, action_id: str = "fe_qt",
    output_distribution: str = "normal",
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Force distribution to normal or uniform."""
    from sklearn.preprocessing import QuantileTransformer
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    vals = pd.to_numeric(df[column], errors="coerce").fillna(0).values.reshape(-1, 1)
    n_quantiles = min(len(df), 1000)
    qt = QuantileTransformer(output_distribution=output_distribution, n_quantiles=n_quantiles)
    df[column] = qt.fit_transform(vals).ravel()
    return df, _record(
        action_id, FeatureEngineeringActionType.QUANTILE_TRANSFORM,
        [column], [], [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Discretization
# ─────────────────────────────────────────────────────────────────────────────

def binning(
    df: pd.DataFrame, column: str, action_id: str = "fe_bin",
    n_bins: int = 5, strategy: str = "quantile",
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Discretize a continuous column into bins. Creates {column}_binned."""
    from sklearn.preprocessing import KBinsDiscretizer
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    vals = pd.to_numeric(df[column], errors="coerce").fillna(0).values.reshape(-1, 1)
    kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy=strategy)
    new_col = f"{column}_binned"
    df[new_col] = kbd.fit_transform(vals).ravel().astype(int)
    return df, _record(
        action_id, FeatureEngineeringActionType.BINNING,
        [column], [new_col], [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature creation
# ─────────────────────────────────────────────────────────────────────────────

def interaction_features(
    df: pd.DataFrame, column_pairs: list, action_id: str = "fe_int",
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Create multiplication interaction features for column pairs."""
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    added = []
    for pair in column_pairs:
        if len(pair) != 2:
            continue
        a, b = pair[0], pair[1]
        if a in df.columns and b in df.columns:
            new_col = f"{a}_x_{b}"
            df[new_col] = pd.to_numeric(df[a], errors="coerce") * pd.to_numeric(df[b], errors="coerce")
            added.append(new_col)
    return df, _record(
        action_id, FeatureEngineeringActionType.INTERACTION_FEATURES,
        [p[0] for p in column_pairs if len(p) == 2] + [p[1] for p in column_pairs if len(p) == 2],
        added, [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


def polynomial_features(
    df: pd.DataFrame, columns: list[str], action_id: str = "fe_poly",
    degree: int = 2, interaction_only: bool = False,
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Create polynomial features for selected columns."""
    from sklearn.preprocessing import PolynomialFeatures
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    valid_cols = [c for c in columns if c in df.columns]
    if not valid_cols:
        raise ValueError(f"None of columns {columns} found in DataFrame")
    sub = df[valid_cols].apply(pd.to_numeric, errors="coerce").fillna(0)
    pf = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    poly_arr = pf.fit_transform(sub)
    feature_names = pf.get_feature_names_out(valid_cols)
    new_cols = [n for n in feature_names if n not in valid_cols]
    poly_df = pd.DataFrame(poly_arr, columns=feature_names, index=df.index)
    for nc in new_cols:
        df[nc] = poly_df[nc]
    return df, _record(
        action_id, FeatureEngineeringActionType.POLYNOMIAL_FEATURES,
        valid_cols, new_cols, [],
        rows_b, len(df), cols_b, len(df.columns), True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection / reduction
# ─────────────────────────────────────────────────────────────────────────────

def drop_low_variance(
    df: pd.DataFrame, action_id: str = "fe_dlv",
    threshold: float = 0.01,
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Drop numeric columns with near-zero variance."""
    from sklearn.feature_selection import VarianceThreshold
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return df, _record(action_id, FeatureEngineeringActionType.DROP_LOW_VARIANCE,
                           [], [], [], rows_b, len(df), cols_b, len(df.columns), True)
    sub = df[numeric_cols].fillna(0)
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(sub)
    dropped = [c for c, keep in zip(numeric_cols, vt.get_support()) if not keep]
    df = df.drop(columns=dropped)
    return df, _record(
        action_id, FeatureEngineeringActionType.DROP_LOW_VARIANCE,
        numeric_cols, [], dropped,
        rows_b, len(df), cols_b, len(df.columns), True,
    )


def drop_high_cardinality(
    df: pd.DataFrame, action_id: str = "fe_dhc",
    max_cardinality: int = 50,
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Drop categorical columns with too many unique values."""
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    dropped = [c for c in cat_cols if df[c].nunique() > max_cardinality]
    df = df.drop(columns=dropped)
    return df, _record(
        action_id, FeatureEngineeringActionType.DROP_HIGH_CARDINALITY,
        cat_cols, [], dropped,
        rows_b, len(df), cols_b, len(df.columns), True,
    )


def drop_highly_correlated(
    df: pd.DataFrame, action_id: str = "fe_dcorr",
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Drop one column from each pair of highly correlated columns."""
    rows_b, cols_b = len(df), len(df.columns)
    df = df.copy()
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] < 2:
        return df, _record(action_id, FeatureEngineeringActionType.DROP_HIGHLY_CORRELATED,
                           [], [], [], rows_b, len(df), cols_b, len(df.columns), True)
    corr = numeric.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped = [col for col in upper.columns if any(upper[col] > threshold)]
    df = df.drop(columns=dropped)
    return df, _record(
        action_id, FeatureEngineeringActionType.DROP_HIGHLY_CORRELATED,
        list(numeric.columns), [], dropped,
        rows_b, len(df), cols_b, len(df.columns), True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher
# ─────────────────────────────────────────────────────────────────────────────

_DISPATCH: dict[FeatureEngineeringActionType, Any] = {
    FeatureEngineeringActionType.ONE_HOT_ENCODE: lambda df, a: one_hot_encode(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.LABEL_ENCODE: lambda df, a: label_encode(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.ORDINAL_ENCODE: lambda df, a: ordinal_encode(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.TARGET_ENCODE: lambda df, a: target_encode(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.FREQUENCY_ENCODE: lambda df, a: frequency_encode(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.MIN_MAX_SCALE: lambda df, a: min_max_scale(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.STANDARD_SCALE: lambda df, a: standard_scale(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.ROBUST_SCALE: lambda df, a: robust_scale(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.MAX_ABS_SCALE: lambda df, a: max_abs_scale(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.LOG_TRANSFORM: lambda df, a: log_transform(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.POWER_TRANSFORM: lambda df, a: power_transform(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.QUANTILE_TRANSFORM: lambda df, a: quantile_transform(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.BINNING: lambda df, a: binning(
        df, a.column_name, a.id, **a.parameters),
    FeatureEngineeringActionType.INTERACTION_FEATURES: lambda df, a: interaction_features(
        df, a.parameters.get("column_pairs", []), a.id),
    FeatureEngineeringActionType.POLYNOMIAL_FEATURES: lambda df, a: polynomial_features(
        df, a.parameters.get("columns", a.columns or []), a.id,
        a.parameters.get("degree", 2), a.parameters.get("interaction_only", False)),
    FeatureEngineeringActionType.DROP_LOW_VARIANCE: lambda df, a: drop_low_variance(
        df, a.id, **{k: v for k, v in a.parameters.items() if k == "threshold"}),
    FeatureEngineeringActionType.DROP_HIGH_CARDINALITY: lambda df, a: drop_high_cardinality(
        df, a.id, **{k: v for k, v in a.parameters.items() if k == "max_cardinality"}),
    FeatureEngineeringActionType.DROP_HIGHLY_CORRELATED: lambda df, a: drop_highly_correlated(
        df, a.id, **{k: v for k, v in a.parameters.items() if k == "threshold"}),
}


def apply_feature_engineering_action(
    df: pd.DataFrame,
    action: FeatureEngineeringAction,
) -> tuple[pd.DataFrame, FeatureEngineeringRecord]:
    """Route a FeatureEngineeringAction to the correct transformation function."""
    fn = _DISPATCH.get(action.action_type)
    if fn is None:
        raise ValueError(f"Unknown action type: {action.action_type}")
    return fn(df, action)
