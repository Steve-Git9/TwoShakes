"""Deduplication utilities — exact and fuzzy."""

from typing import Optional
import pandas as pd
from thefuzz import fuzz  # type: ignore


def remove_exact_duplicates(
    df: pd.DataFrame,
    keep: str = "first",
    subset: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Remove exact duplicate rows.

    Args:
        keep: "first" | "last" | False
        subset: list of columns to consider; None = all columns

    Returns:
        (deduplicated_df, rows_removed)
    """
    df = df.copy()
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
    return df, before - len(df)


def remove_fuzzy_duplicates(
    df: pd.DataFrame,
    threshold: int = 90,
    key_columns: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Remove near-duplicate rows where concatenated string similarity >= threshold.

    Keeps the first occurrence of each fuzzy-duplicate group.

    Returns:
        (deduplicated_df, rows_removed)
    """
    df = df.copy()
    before = len(df)

    cols = key_columns if key_columns else df.select_dtypes(include="object").columns.tolist()
    if not cols:
        return df, 0

    concat = df[cols].fillna("").apply(lambda r: " ".join(str(v) for v in r.values), axis=1)

    keep_mask = [True] * len(df)
    for i in range(len(concat)):
        if not keep_mask[i]:
            continue
        for j in range(i + 1, len(concat)):
            if not keep_mask[j]:
                continue
            if fuzz.ratio(concat.iloc[i], concat.iloc[j]) >= threshold:
                keep_mask[j] = False

    df = df[keep_mask].reset_index(drop=True)
    return df, before - len(df)
