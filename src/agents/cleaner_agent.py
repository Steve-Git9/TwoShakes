"""
Cleaner Agent — executes approved CleaningActions as deterministic pandas transforms.
The LLM never touches the data directly.
"""

import logging
from typing import Any

import pandas as pd

from src.models.schemas import (
    ActionType,
    CleaningAction,
    CleaningPlan,
    TransformationLog,
    TransformationRecord,
)

logger = logging.getLogger(__name__)


def _sample(series: pd.Series, n: int = 3) -> list[Any]:
    """Return up to n non-null values as plain Python strings."""
    return [str(v) for v in series.dropna().head(n).tolist()]


async def execute_cleaning_plan(
    df: pd.DataFrame,
    plan: CleaningPlan,
) -> tuple[pd.DataFrame, TransformationLog]:
    """
    Execute all approved actions in the CleaningPlan.

    Each action is mapped to its deterministic transformation function.
    Failures are logged and skipped — the pipeline never crashes.

    Returns:
        (cleaned_df, TransformationLog)
    """
    records: list[TransformationRecord] = []
    total_rows_modified = 0

    for action in plan.actions:
        if not action.approved:
            continue

        col = action.column_name
        before_col = df[col].copy() if col and col in df.columns else None
        success = True
        error_msg = None
        rows_affected = 0

        try:
            df, rows_affected = _dispatch(df, action)
        except Exception as e:
            success = False
            error_msg = str(e)
            logger.error(
                f"[{action.id}] {action.action_type.value} on '{col or 'all rows'}' failed: {e}"
            )

        after_col = df[col] if col and col in df.columns else None

        record = TransformationRecord(
            action_id=action.id,
            column_name=col,
            action_type=action.action_type,
            rows_affected=rows_affected,
            before_sample=_sample(before_col) if before_col is not None else [],
            after_sample=_sample(after_col) if after_col is not None else [],
            success=success,
            error_message=error_msg,
        )
        records.append(record)
        total_rows_modified += rows_affected

        logger.info(
            f"[{action.id}] {action.action_type.value} on '{col or 'all rows'}' "
            f"— {'OK' if success else 'FAIL'} ({rows_affected} rows affected)"
        )

    log = TransformationLog(
        records=records,
        total_actions_executed=len(records),
        total_actions_succeeded=sum(1 for r in records if r.success),
        total_rows_modified=total_rows_modified,
    )
    return df, log


def _dispatch(df: pd.DataFrame, action: CleaningAction) -> tuple[pd.DataFrame, int]:
    """Map an ActionType to its transformation function and execute it."""
    col = action.column_name
    p = action.parameters
    t = action.action_type

    if t == ActionType.TRIM_WHITESPACE:
        from src.transformations.text_cleaning import trim_whitespace
        return trim_whitespace(df, col)

    elif t == ActionType.NORMALIZE_CASE:
        from src.transformations.text_cleaning import normalize_case
        return normalize_case(df, col, p.get("case", "lower"))

    elif t == ActionType.FIX_ENCODING:
        from src.transformations.text_cleaning import fix_encoding
        return fix_encoding(df, col)

    elif t == ActionType.CONVERT_TYPE:
        from src.transformations.type_conversion import convert_column_type
        return convert_column_type(df, col, p.get("target_type", "str"), p.get("remove_chars"))

    elif t == ActionType.PARSE_DATES:
        from src.transformations.datetime_parser import parse_dates
        return parse_dates(df, col, p.get("target_format", "%Y-%m-%d"))

    elif t == ActionType.FILL_MISSING:
        from src.transformations.missing_values import fill_missing
        return fill_missing(df, col, p.get("strategy", "mode"), p.get("value"))

    elif t == ActionType.STANDARDIZE_CATEGORICAL:
        mapping = p.get("mapping", {})
        if mapping:
            from src.transformations.categorical import standardize_categories
            return standardize_categories(df, col, mapping)
        else:
            from src.transformations.categorical import auto_standardize
            return auto_standardize(df, col, p.get("threshold", 80))

    elif t == ActionType.HANDLE_OUTLIER:
        from src.transformations.outliers import handle_outliers_iqr
        return handle_outliers_iqr(
            df, col,
            action=p.get("action", "cap"),
            multiplier=float(p.get("multiplier", 1.5)),
        )

    elif t == ActionType.REMOVE_DUPLICATES:
        subset = p.get("subset") or None
        if isinstance(subset, list) and len(subset) == 0:
            subset = None
        from src.transformations.deduplication import remove_exact_duplicates
        return remove_exact_duplicates(df, keep=p.get("keep", "first"), subset=subset)

    elif t == ActionType.RENAME_COLUMN:
        if col and col in df.columns:
            new_name = p.get("new_name", col)
            df = df.copy().rename(columns={col: new_name})
            return df, 1
        return df, 0

    elif t == ActionType.DROP_COLUMN:
        if col and col in df.columns:
            df = df.copy().drop(columns=[col])
            return df, 1
        return df, 0

    else:
        logger.warning(f"Unknown action type: {t}")
        return df, 0
