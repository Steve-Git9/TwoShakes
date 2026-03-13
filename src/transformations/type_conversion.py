"""Type conversion utilities."""

from typing import Optional
import pandas as pd


def convert_column_type(
    df: pd.DataFrame,
    column: str,
    target_type: str,
    remove_chars: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, int]:
    """
    Convert a column to target_type, optionally stripping characters first.

    Supported target_type values: float, numeric, int, str, bool.
    Unconvertible per-cell values are set to NaN instead of raising.

    Returns:
        (modified_df, rows_affected)
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0

    before = df[column].copy()

    # Strip unwanted characters (e.g. "$", ",")
    if remove_chars:
        pattern = "[" + "".join(map(re.escape, remove_chars)) + "]"
        series = df[column].astype(str).str.strip().str.replace(pattern, "", regex=True)
    else:
        series = df[column].astype(str).str.strip()

    if target_type in ("float", "numeric"):
        # Also handle currency chars even if not explicitly listed
        series = series.str.replace(r"[$,\s]", "", regex=True)
        df[column] = pd.to_numeric(series, errors="coerce")
    elif target_type == "int":
        series = series.str.replace(r"[$,\s]", "", regex=True)
        df[column] = pd.to_numeric(series, errors="coerce").round().astype("Int64")
    elif target_type == "str":
        df[column] = series
    elif target_type == "bool":
        mapping = {
            "true": True, "false": False,
            "yes": True, "no": False,
            "1": True, "0": False,
            "t": True, "f": False,
        }
        df[column] = series.str.lower().map(mapping)
    else:
        return df, 0

    rows_affected = int((df[column] != before).sum())
    return df, rows_affected


# Needed for the remove_chars pattern escaping
import re  # noqa: E402 (imported at end intentionally to keep function signature readable)
