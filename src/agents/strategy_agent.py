"""
Strategy Agent — takes a ProfileReport + raw DataFrame and produces a CleaningPlan via the LLM.
"""

import json
import logging
from datetime import datetime

import pandas as pd

from src.agents import AgentClient
from src.models.schemas import (
    ActionType,
    CleaningAction,
    CleaningPlan,
    Priority,
    ProfileReport,
)

logger = logging.getLogger(__name__)

_STRATEGY_INSTRUCTIONS = """\
You are a data cleaning strategist. You receive a data profiling report and must generate a precise cleaning plan.

For each issue, propose a CleaningAction with these exact fields:
- id: unique string like "action_001", "action_002", etc.
- column_name: target column name (null for row-level actions like removing duplicates)
- action_type: EXACTLY one of: convert_type, fill_missing, standardize_categorical, parse_dates, remove_duplicates, handle_outlier, fix_encoding, rename_column, drop_column, trim_whitespace, normalize_case
- description: human-readable explanation of what this does
- parameters: specific config for the transformation. Examples:
    - convert_type: {"target_type": "float", "remove_chars": ["$", ","]}
    - fill_missing: {"strategy": "median"} or {"strategy": "mode"} or {"strategy": "custom", "value": 0}
    - standardize_categorical: {"mapping": {"USA": "US", "United States": "US", "U.S.": "US"}}
    - parse_dates: {"target_format": "%Y-%m-%d"}
    - remove_duplicates: {"keep": "first", "subset": null}
    - handle_outlier: {"method": "iqr", "action": "cap"} or {"method": "iqr", "action": "remove"}
    - trim_whitespace: {}
    - normalize_case: {"case": "title"} or {"case": "lower"}
    - rename_column: {"new_name": "clean_name"}
- reason: why this action is needed
- priority: "high", "medium", or "low"

Order actions logically:
1. trim_whitespace and fix_encoding first
2. type conversions and date parsing
3. missing value handling
4. categorical standardization
5. outlier handling
6. deduplication last

Respond ONLY in valid JSON: {"actions": [...], "estimated_rows_affected": number}
"""


async def generate_cleaning_plan(profile: ProfileReport, df: pd.DataFrame) -> CleaningPlan:
    """
    Generate a CleaningPlan from a ProfileReport + raw DataFrame.
    Sends the profile JSON to the LLM; parses the response into a CleaningPlan.
    """
    client = AgentClient(
        name="StrategyAgent",
        instructions=_STRATEGY_INSTRUCTIONS,
        json_mode=True,
    )

    # Build a compact payload: profile JSON + column head values for extra context
    profile_dict = json.loads(profile.model_dump_json())
    for col_profile in profile_dict.get("columns", []):
        col_name = col_profile["column_name"]
        if col_name in df.columns:
            col_profile["head_values"] = df[col_name].dropna().head(5).tolist()

    payload = json.dumps(profile_dict, default=str)

    raw = await client.run(payload)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning(f"StrategyAgent returned invalid JSON ({e}); returning empty plan")
        data = {"actions": [], "estimated_rows_affected": 0}

    actions: list[CleaningAction] = []
    seen_ids: set[str] = set()

    for i, a in enumerate(data.get("actions", [])):
        try:
            action_id = str(a.get("id") or "").strip() or f"action_{i+1:03d}"
            if action_id in seen_ids:
                action_id = f"{action_id}_{i}"
            seen_ids.add(action_id)

            actions.append(CleaningAction(
                id=action_id,
                column_name=a.get("column_name") or None,
                action_type=ActionType(a["action_type"]),
                description=a.get("description", ""),
                parameters=a.get("parameters") or {},
                reason=a.get("reason", ""),
                priority=Priority(a.get("priority", "medium")),
                approved=True,
            ))
        except Exception as e:
            logger.warning(f"Skipping invalid action at index {i}: {e} — raw: {a}")

    return CleaningPlan(
        actions=actions,
        estimated_rows_affected=data.get("estimated_rows_affected", 0),
        generated_at=datetime.utcnow(),
    )
