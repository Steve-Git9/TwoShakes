"""Categorical value standardisation."""

import pandas as pd
from thefuzz import fuzz, process  # type: ignore


def standardize_categories(
    df: pd.DataFrame,
    column: str,
    mapping: dict,
) -> tuple[pd.DataFrame, int]:
    """
    Apply an explicit {raw_value: canonical_value} mapping to a column.

    Returns:
        (modified_df, rows_affected)
    """
    df = df.copy()
    if column not in df.columns or not mapping:
        return df, 0

    before = df[column].copy()
    df[column] = df[column].astype(str).str.strip().replace(mapping)
    rows_affected = int((df[column] != before).sum())
    return df, rows_affected


def auto_standardize(
    df: pd.DataFrame,
    column: str,
    threshold: int = 80,
) -> tuple[pd.DataFrame, int]:
    """
    Automatically merge similar categorical values above `threshold` similarity
    into the most frequent variant using thefuzz.

    Returns:
        (modified_df, rows_affected)
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0

    values = df[column].astype(str).str.strip()
    vc = values.value_counts()
    if len(vc) == 0:
        return df, 0

    choices = vc.index.tolist()  # sorted by frequency desc
    auto_map: dict[str, str] = {}

    for val in choices:
        if val in auto_map:
            continue
        # Find the best match among more-frequent values (already canonical)
        more_freq = [c for c in choices if c not in auto_map and vc[c] >= vc[val] and c != val]
        if not more_freq:
            continue
        match, score = process.extractOne(val, more_freq, scorer=fuzz.token_sort_ratio)
        if score >= threshold:
            auto_map[val] = match

    if not auto_map:
        return df, 0

    before = df[column].copy()
    df[column] = values.replace(auto_map)
    rows_affected = int((df[column] != before).sum())
    return df, rows_affected
