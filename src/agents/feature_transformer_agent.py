"""
Feature Transformer Agent
===========================
Executes an approved FeatureEngineeringPlan on a cleaned DataFrame.
Returns the ML-ready DataFrame and a FeatureEngineeringLog.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.models.schemas import (
    FeatureEngineeringLog,
    FeatureEngineeringPlan,
    FeatureEngineeringRecord,
)
from src.transformations.feature_engineering import apply_feature_engineering_action

logger = logging.getLogger(__name__)


async def execute_feature_engineering(
    df: pd.DataFrame,
    plan: FeatureEngineeringPlan,
) -> tuple[pd.DataFrame, FeatureEngineeringLog]:
    """
    Execute all approved actions in a FeatureEngineeringPlan.

    Args:
        df:   Cleaned DataFrame (result of the cleaning pipeline).
        plan: Feature engineering plan with user-approved actions.

    Returns:
        (ml_ready_df, FeatureEngineeringLog)
    """
    approved_actions = [a for a in plan.actions if a.approved]
    logger.info(
        f"[FeatureTransformerAgent] Executing {len(approved_actions)} / "
        f"{len(plan.actions)} actions on {len(df)} rows × {len(df.columns)} cols"
    )

    records: list[FeatureEngineeringRecord] = []
    current_df = df.copy()

    for action in approved_actions:
        logger.info(
            f"[FeatureTransformerAgent] {action.action_type.value} "
            f"on {action.column_name or action.columns}"
        )
        try:
            current_df, record = apply_feature_engineering_action(current_df, action)
            records.append(record)
            logger.info(
                f"[FeatureTransformerAgent]   ✓ {action.id}: "
                f"+{len(record.columns_added)} cols, "
                f"-{len(record.columns_removed)} cols"
            )
        except Exception as e:
            logger.error(
                f"[FeatureTransformerAgent]   ✗ {action.id} failed: {e}"
            )
            # Record failure but continue
            records.append(
                FeatureEngineeringRecord(
                    action_id=action.id,
                    action_type=action.action_type,
                    columns_affected=[action.column_name or ""] if action.column_name else (action.columns or []),
                    columns_added=[],
                    columns_removed=[],
                    rows_before=len(current_df),
                    rows_after=len(current_df),
                    cols_before=len(current_df.columns),
                    cols_after=len(current_df.columns),
                    success=False,
                    error_message=str(e),
                )
            )

    succeeded = [r for r in records if r.success]
    total_added = sum(len(r.columns_added) for r in records if r.success)
    total_removed = sum(len(r.columns_removed) for r in records if r.success)

    log = FeatureEngineeringLog(
        records=records,
        total_actions_executed=len(records),
        total_actions_succeeded=len(succeeded),
        columns_added_total=total_added,
        columns_removed_total=total_removed,
    )

    logger.info(
        f"[FeatureTransformerAgent] Done: "
        f"{len(succeeded)}/{len(records)} succeeded, "
        f"+{total_added}/-{total_removed} cols, "
        f"final shape: {len(current_df)} × {len(current_df.columns)}"
    )
    return current_df, log
