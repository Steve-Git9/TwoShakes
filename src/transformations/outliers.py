"""Outlier detection and handling using the IQR method."""

import pandas as pd


def handle_outliers_iqr(
    df: pd.DataFrame,
    column: str,
    action: str = "cap",
    multiplier: float = 1.5,
) -> tuple[pd.DataFrame, int]:
    """
    Handle outliers in a numeric column using the IQR method.

    Args:
        action: "cap" — clip to [Q1-k*IQR, Q3+k*IQR]
                "remove" — drop outlier rows
                "flag" — set outlier cells to NaN

    Returns:
        (modified_df, rows_affected)
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0

    numeric = pd.to_numeric(df[column], errors="coerce")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    outlier_mask = (numeric < lower) | (numeric > upper)
    rows_affected = int(outlier_mask.sum())

    if rows_affected == 0:
        return df, 0

    if action == "cap":
        df[column] = numeric.clip(lower=lower, upper=upper)
    elif action == "remove":
        df = df[~outlier_mask].reset_index(drop=True)
    elif action == "flag":
        df.loc[outlier_mask, column] = pd.NA

    return df, rows_affected
