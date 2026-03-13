"""Missing value standardisation and imputation."""

from typing import Any, Optional
import pandas as pd

DEFAULT_MISSING_INDICATORS = [
    "N/A", "NA", "n/a", "na", "null", "NULL", "Null",
    "none", "None", "NONE", "", "-", ".", "999", "-999",
]


def standardize_missing(
    df: pd.DataFrame,
    column: str,
    missing_indicators: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Replace all known missing-value sentinels in a column with actual NaN.

    Returns:
        (modified_df, rows_affected)
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0

    indicators = missing_indicators if missing_indicators is not None else DEFAULT_MISSING_INDICATORS
    before_null = df[column].isna().sum()
    df[column] = df[column].astype(str).str.strip()
    df[column] = df[column].replace(indicators, pd.NA)
    after_null = df[column].isna().sum()
    return df, int(after_null - before_null)


def fill_missing(
    df: pd.DataFrame,
    column: str,
    strategy: str = "mode",
    custom_value: Optional[Any] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Fill NaN values in a column using the chosen strategy.

    Strategies: mean, median, mode, ffill, bfill, custom, drop.

    Returns:
        (modified_df, rows_affected)
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0

    # Normalise sentinels first
    df, _ = standardize_missing(df, column)

    before_null = int(df[column].isna().sum())
    if before_null == 0:
        return df, 0

    if strategy == "mean":
        val = pd.to_numeric(df[column], errors="coerce").mean()
        df[column] = df[column].fillna(val)
    elif strategy == "median":
        val = pd.to_numeric(df[column], errors="coerce").median()
        df[column] = df[column].fillna(val)
    elif strategy == "mode":
        mode = df[column].mode()
        if not mode.empty:
            df[column] = df[column].fillna(mode.iloc[0])
    elif strategy in ("custom", "constant"):
        df[column] = df[column].fillna(custom_value)
    elif strategy == "ffill":
        df[column] = df[column].ffill()
    elif strategy == "bfill":
        df[column] = df[column].bfill()
    elif strategy == "drop":
        before_len = len(df)
        df = df.dropna(subset=[column]).reset_index(drop=True)
        return df, before_len - len(df)

    after_null = int(df[column].isna().sum())
    return df, before_null - after_null
