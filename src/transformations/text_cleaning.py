"""Text cleaning utilities."""

import re
import pandas as pd

try:
    import ftfy as _ftfy  # type: ignore
    _FTFY_AVAILABLE = True
except ImportError:
    _FTFY_AVAILABLE = False


def trim_whitespace(
    df: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, int]:
    """
    Strip leading/trailing whitespace from every cell in a column.

    Returns:
        (modified_df, rows_affected)
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0

    before = df[column].copy()
    df[column] = df[column].astype(str).str.strip()
    rows_affected = int((df[column] != before).sum())
    return df, rows_affected


def normalize_case(
    df: pd.DataFrame,
    column: str,
    case: str = "lower",
) -> tuple[pd.DataFrame, int]:
    """
    Normalise string case in a column.

    Args:
        case: "lower" | "upper" | "title"

    Returns:
        (modified_df, rows_affected)
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0

    before = df[column].copy()
    if case == "lower":
        df[column] = df[column].astype(str).str.lower()
    elif case == "upper":
        df[column] = df[column].astype(str).str.upper()
    elif case == "title":
        df[column] = df[column].astype(str).str.title()

    rows_affected = int((df[column] != before).sum())
    return df, rows_affected


def fix_encoding(
    df: pd.DataFrame,
    column: str,
) -> tuple[pd.DataFrame, int]:
    """
    Fix mojibake (garbled encoding) in a text column using ftfy.
    Falls back to a no-op if ftfy is not installed.

    Returns:
        (modified_df, rows_affected)
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0

    before = df[column].copy()
    if not _FTFY_AVAILABLE:
        return df, 0
    try:
        df[column] = df[column].astype(str).apply(_ftfy.fix_text)
    except Exception:
        return df, 0

    rows_affected = int((df[column] != before).sum())
    return df, rows_affected
