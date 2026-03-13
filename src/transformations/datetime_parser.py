"""Date/time parsing and normalisation."""

import pandas as pd
from dateutil.parser import parse as dateutil_parse, ParserError

_MISSING = {"", "N/A", "NA", "n/a", "null", "NULL", "-", "none", "None", "NaT", "nat"}


def parse_dates(
    df: pd.DataFrame,
    column: str,
    target_format: str = "%Y-%m-%d",
) -> tuple[pd.DataFrame, int]:
    """
    Parse mixed-format date strings in a column to a uniform format.

    Unrecognised values become NaN. Returns (modified_df, rows_affected).
    """
    df = df.copy()
    if column not in df.columns:
        return df, 0

    before = df[column].copy()

    def _safe_parse(val):
        s = str(val).strip()
        if pd.isna(val) or s in _MISSING:
            return pd.NaT
        try:
            return dateutil_parse(s, fuzzy=True)
        except (ParserError, ValueError, OverflowError, TypeError):
            return pd.NaT

    parsed = df[column].apply(_safe_parse)
    df[column] = parsed.dt.strftime(target_format)

    rows_affected = int((df[column] != before).sum())
    return df, rows_affected
