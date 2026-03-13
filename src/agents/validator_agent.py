"""
Validator Agent — runs Python quality checks on cleaned data, then calls the LLM
for a professional "Data Quality Report Card".
"""

import json
import logging

import pandas as pd

from src.agents import AgentClient
from src.models.schemas import (
    ProfileReport,
    TransformationLog,
    ValidationCheck,
    ValidationReport,
)

logger = logging.getLogger(__name__)

_VALIDATOR_INSTRUCTIONS = """\
You are a data quality auditor. You receive before/after statistics for a dataset \
that went through automated cleaning.

Generate a quality report card with:
1. quality_score: 0-100 (integer)
2. certificate: a 3-5 sentence professional summary of improvements and remaining notes
3. remaining_concerns: list of any issues still present
4. analysis_ready: boolean — is this data ready for analysis?

Be honest. If issues remain, say so.
Respond ONLY in valid JSON:
{
  "quality_score": 0,
  "certificate": "string",
  "remaining_concerns": [],
  "analysis_ready": true
}
"""


def _compute_after_score(df_before: pd.DataFrame, df_after: pd.DataFrame, before_score: float) -> float:
    """
    Compute an after-cleaning quality score using a weighted formula:
    - Null reduction contributes 40 points
    - Duplicate removal contributes 20 points
    - Baseline carries forward at 40 points
    """
    null_pct_before = df_before.isna().sum().sum() / max(df_before.size, 1) * 100
    null_pct_after = df_after.isna().sum().sum() / max(df_after.size, 1) * 100
    null_improvement = max(0.0, null_pct_before - null_pct_after)

    dup_before = int(df_before.duplicated().sum())
    dup_after = int(df_after.duplicated().sum())
    dup_improvement = max(0.0, dup_before - dup_after)

    score = before_score
    score += min(null_improvement * 2, 15)        # up to +15 for null reduction
    score += min(dup_improvement * 0.5, 10)        # up to +10 for dedup
    return round(min(score, 100.0), 2)


async def validate(
    original_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
    profile_before: ProfileReport,
    transformation_log: TransformationLog,
) -> ValidationReport:
    """
    Run deterministic quality checks + LLM report card.

    Returns:
        ValidationReport
    """
    checks: list[ValidationCheck] = []

    # --- Check 1: Row count preservation ---
    row_diff = len(original_df) - len(cleaned_df)
    row_loss_pct = row_diff / max(len(original_df), 1) * 100
    checks.append(ValidationCheck(
        check_name="Row count",
        passed=len(cleaned_df) > 0 and row_loss_pct < 20,
        details=f"{len(original_df)} -> {len(cleaned_df)} rows ({row_diff} removed, {row_loss_pct:.1f}% loss)",
        severity="warning" if row_loss_pct >= 10 else "info",
    ))

    # --- Check 2: No exact duplicates remain ---
    dup_after = int(cleaned_df.duplicated().sum())
    checks.append(ValidationCheck(
        check_name="No exact duplicates",
        passed=dup_after == 0,
        details=f"{dup_after} duplicate rows remain" if dup_after else "No duplicates found",
        severity="warning" if dup_after > 0 else "info",
    ))

    # --- Check 3: Missing value reduction ---
    null_pct_before = original_df.isna().sum().sum() / max(original_df.size, 1) * 100
    null_pct_after = cleaned_df.isna().sum().sum() / max(cleaned_df.size, 1) * 100
    null_improved = null_pct_after <= null_pct_before
    checks.append(ValidationCheck(
        check_name="Missing value reduction",
        passed=null_improved,
        details=f"Null %: {null_pct_before:.1f}% -> {null_pct_after:.1f}%",
        severity="info" if null_improved else "warning",
    ))

    # --- Check 4: No all-null columns ---
    null_cols = [c for c in cleaned_df.columns if cleaned_df[c].isna().all()]
    checks.append(ValidationCheck(
        check_name="No all-null columns",
        passed=len(null_cols) == 0,
        details=f"All-null columns: {null_cols}" if null_cols else "None",
        severity="warning" if null_cols else "info",
    ))

    # --- Check 5: Column type consistency (no mixed types) ---
    mixed_cols = []
    for col in cleaned_df.columns:
        non_null = cleaned_df[col].dropna()
        if len(non_null) == 0:
            continue
        types = set(type(v).__name__ for v in non_null.head(50))
        if len(types) > 1:
            mixed_cols.append(f"{col}({','.join(types)})")
    checks.append(ValidationCheck(
        check_name="Column type consistency",
        passed=len(mixed_cols) == 0,
        details=f"Mixed-type columns: {mixed_cols}" if mixed_cols else "All columns are type-consistent",
        severity="warning" if mixed_cols else "info",
    ))

    # --- Check 6: Transformation success rate ---
    success_rate = (
        transformation_log.total_actions_succeeded / transformation_log.total_actions_executed
        if transformation_log.total_actions_executed > 0 else 1.0
    )
    checks.append(ValidationCheck(
        check_name="Transformation success rate",
        passed=success_rate >= 0.9,
        details=(
            f"{transformation_log.total_actions_succeeded}/"
            f"{transformation_log.total_actions_executed} actions succeeded "
            f"({success_rate*100:.0f}%)"
        ),
        severity="warning" if success_rate < 0.9 else "info",
    ))

    before_score = profile_before.overall_quality_score
    after_score = _compute_after_score(original_df, cleaned_df, before_score)
    improvement = round(after_score - before_score, 2)

    # --- LLM Report Card ---
    client = AgentClient(
        name="ValidatorAgent",
        instructions=_VALIDATOR_INSTRUCTIONS,
        json_mode=True,
    )
    payload = json.dumps({
        "before_score": before_score,
        "after_score": after_score,
        "improvement": improvement,
        "row_count_before": len(original_df),
        "row_count_after": len(cleaned_df),
        "null_pct_before": round(null_pct_before, 2),
        "null_pct_after": round(null_pct_after, 2),
        "duplicates_before": int(original_df.duplicated().sum()),
        "duplicates_after": dup_after,
        "transformation_summary": {
            "actions_executed": transformation_log.total_actions_executed,
            "actions_succeeded": transformation_log.total_actions_succeeded,
            "rows_modified": transformation_log.total_rows_modified,
        },
        "checks": [c.model_dump() for c in checks],
    })

    raw = await client.run(payload)
    try:
        card = json.loads(raw)
        certificate = card.get("certificate", "Cleaning completed.")
        llm_score = card.get("quality_score")
        if llm_score is not None:
            after_score = round(float(llm_score), 2)
            improvement = round(after_score - before_score, 2)
    except Exception as e:
        logger.warning(f"ValidatorAgent JSON parse failed: {e}")
        certificate = "Data cleaning pipeline completed successfully."

    return ValidationReport(
        checks=checks,
        before_quality_score=before_score,
        after_quality_score=after_score,
        improvement_percentage=improvement,
        data_quality_certificate=certificate,
        transformation_log=transformation_log,
        row_count_before=len(original_df),
        row_count_after=len(cleaned_df),
        col_count_before=len(original_df.columns),
        col_count_after=len(cleaned_df.columns),
    )
