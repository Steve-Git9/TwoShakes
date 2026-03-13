"""
DataPrepAgent MCP Server
========================
Exposes the full data-preparation + feature-engineering pipeline as Model
Context Protocol (MCP) tools so any MCP client — GitHub Copilot Agent Mode,
Claude Desktop, VS Code Copilot Chat — can profile, plan, clean, validate,
and ML-prepare datasets without a UI.

Usage
-----
Run the server (stdio transport, compatible with all MCP clients):

    python src/mcp_server.py

Add to your MCP client config (e.g. Claude Desktop ~/claude_desktop_config.json):

    {
      "mcpServers": {
        "dataprepagent": {
          "command": "python",
          "args": ["<absolute-path-to-repo>/src/mcp_server.py"],
          "env": {
            "AZURE_AI_PROJECT_ENDPOINT": "...",
            "AZURE_AI_PROJECT_KEY": "...",
            "AZURE_AI_MODEL_DEPLOYMENT_NAME": "gpt-4o"
          }
        }
      }
    }

Tools exposed
-------------
Cleaning pipeline:
1. profile_data(file_path)                          → ProfileReport JSON
2. suggest_cleaning_plan(file_path, profile_json)   → CleaningPlan JSON
3. clean_data(file_path, plan_json)                 → cleaned CSV path + TransformationLog JSON
4. validate_cleaning(original, cleaned, profile, log) → ValidationReport JSON
5. list_supported_formats()                         → list of supported file extensions

Feature engineering pipeline:
6. recommend_feature_engineering(cleaned_path, target_column?) → FeatureEngineeringPlan JSON
7. apply_feature_engineering(cleaned_path, fe_plan_json)       → ML-ready CSV path + FeatureEngineeringLog JSON
"""

import sys
import os
import asyncio
import json
import tempfile

# Ensure src/ is importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "DataPrepAgent",
    instructions=(
        "DataPrepAgent automates data preparation and ML feature engineering using "
        "a multi-agent AI pipeline. "
        "Cleaning flow: profile_data → suggest_cleaning_plan → clean_data → validate_cleaning. "
        "Feature engineering flow (optional): recommend_feature_engineering → apply_feature_engineering. "
        "Each tool accepts and returns JSON strings."
    ),
)


# ── Tool 1: profile_data ──────────────────────────────────────────────────────

@mcp.tool()
async def profile_data(file_path: str) -> str:
    """
    Ingest a CSV, Excel, JSON, or XML file and return a full data-quality
    profile as a JSON string (ProfileReport schema).

    Args:
        file_path: Absolute or relative path to the data file.

    Returns:
        JSON string containing the ProfileReport, including column types,
        missing-value counts, outlier flags, duplicate rows, quality score,
        key issues, and an AI-generated summary.
    """
    from src.agents.ingestion_agent import ingest
    from src.agents.profiler_agent import profile_dataframe

    df, metadata = await ingest(file_path)
    report = await profile_dataframe(df, metadata)
    return report.model_dump_json(indent=2)


# ── Tool 2: suggest_cleaning_plan ────────────────────────────────────────────

@mcp.tool()
async def suggest_cleaning_plan(file_path: str, profile_json: str) -> str:
    """
    Given a data file and its ProfileReport JSON (from profile_data), return
    an AI-generated cleaning plan as a JSON string (CleaningPlan schema).

    Args:
        file_path:    Path to the original data file (needed for sample values).
        profile_json: JSON string returned by profile_data.

    Returns:
        JSON string containing the CleaningPlan — an ordered list of
        CleaningActions with priorities, parameters, and reasoning.
        All actions start with approved=true; callers may set approved=false
        to skip specific actions before passing to clean_data.
    """
    from src.agents.ingestion_agent import ingest
    from src.agents.strategy_agent import generate_cleaning_plan
    from src.models.schemas import ProfileReport

    profile = ProfileReport.model_validate_json(profile_json)
    df, _ = await ingest(file_path)
    plan = await generate_cleaning_plan(profile, df)
    return plan.model_dump_json(indent=2)


# ── Tool 3: clean_data ───────────────────────────────────────────────────────

@mcp.tool()
async def clean_data(file_path: str, plan_json: str) -> str:
    """
    Execute a cleaning plan (from suggest_cleaning_plan) on a data file.
    Only actions with approved=true are executed.

    Args:
        file_path: Path to the original data file.
        plan_json: JSON string (CleaningPlan) from suggest_cleaning_plan.
                   You may set any action's "approved" field to false to skip it.

    Returns:
        JSON string with two keys:
          "cleaned_file": path to the cleaned CSV (written next to the original)
          "transformation_log": TransformationLog JSON with per-action results
    """
    from src.agents.ingestion_agent import ingest
    from src.agents.cleaner_agent import execute_cleaning_plan
    from src.models.schemas import CleaningPlan

    df, _ = await ingest(file_path)
    plan = CleaningPlan.model_validate_json(plan_json)

    cleaned_df, tlog = await execute_cleaning_plan(df, plan)

    # Write cleaned file alongside the original
    base, ext = os.path.splitext(file_path)
    cleaned_path = base + "_cleaned.csv"
    cleaned_df.to_csv(cleaned_path, index=False)

    return json.dumps({
        "cleaned_file": cleaned_path,
        "transformation_log": json.loads(tlog.model_dump_json()),
    }, indent=2)


# ── Tool 4: validate_cleaning ────────────────────────────────────────────────

@mcp.tool()
async def validate_cleaning(
    original_path: str,
    cleaned_path: str,
    profile_json: str,
    transformation_log_json: str,
) -> str:
    """
    Compare the original and cleaned files and return a ValidationReport.

    Args:
        original_path:          Path to the original (uncleaned) file.
        cleaned_path:           Path to the cleaned CSV (from clean_data).
        profile_json:           ProfileReport JSON from profile_data.
        transformation_log_json: TransformationLog JSON from clean_data.

    Returns:
        JSON string (ValidationReport) with 6 quality checks, before/after
        quality scores, improvement percentage, and an AI-generated certificate.
    """
    from src.agents.ingestion_agent import ingest
    from src.agents.validator_agent import validate
    from src.models.schemas import ProfileReport, TransformationLog
    import pandas as pd

    original_df, _ = await ingest(original_path)
    cleaned_df = pd.read_csv(cleaned_path, dtype=str)
    profile = ProfileReport.model_validate_json(profile_json)
    tlog    = TransformationLog.model_validate_json(transformation_log_json)

    report = await validate(original_df, cleaned_df, profile, tlog)
    return report.model_dump_json(indent=2)


# ── Tool 5: list_supported_formats ───────────────────────────────────────────

@mcp.tool()
def list_supported_formats() -> str:
    """
    Return the list of file formats DataPrepAgent can ingest.

    Returns:
        JSON array of supported file extensions.
    """
    return json.dumps([".csv", ".tsv", ".xlsx", ".xls", ".json", ".xml", ".pdf"])


# ── Tool 6: recommend_feature_engineering ────────────────────────────────────

@mcp.tool()
async def recommend_feature_engineering(
    cleaned_path: str,
    target_column: str = "",
) -> str:
    """
    Analyze a cleaned dataset and return an AI-generated feature engineering plan.

    Performs two-phase analysis:
    1. Statistical: skewness, cardinality, correlation, variance analysis
    2. LLM reasoning: recommends encoding, scaling, distribution transforms,
       feature creation, and feature selection actions

    Args:
        cleaned_path:  Path to a cleaned CSV file (output of clean_data).
        target_column: Optional name of the prediction target column.
                       Enables target-aware encoding (e.g. target_encode).
                       Pass empty string for general-purpose ML preparation.

    Returns:
        JSON string (FeatureEngineeringPlan) with a list of FeatureEngineeringAction
        objects. Each action has: id, action_type, column_name, description, reason,
        impact, priority, parameters, approved=true, and an optional warning.
        Callers may set any action's "approved" field to false before passing to
        apply_feature_engineering to skip it.
    """
    import pandas as pd
    from src.agents.feature_engineering_agent import recommend_feature_engineering as _rfe

    df = pd.read_csv(cleaned_path)
    target = target_column.strip() or None
    plan = await _rfe(df, target_column=target)
    return plan.model_dump_json(indent=2)


# ── Tool 7: apply_feature_engineering ────────────────────────────────────────

@mcp.tool()
async def apply_feature_engineering(
    cleaned_path: str,
    fe_plan_json: str,
) -> str:
    """
    Execute an approved feature engineering plan on a cleaned dataset.

    Only actions with approved=true are executed. Each action is fault-tolerant:
    failures are logged but do not abort the pipeline.

    Args:
        cleaned_path:  Path to the cleaned CSV (from clean_data).
        fe_plan_json:  JSON string (FeatureEngineeringPlan) from
                       recommend_feature_engineering. You may set any action's
                       "approved" field to false to skip it.

    Returns:
        JSON string with two keys:
          "ml_ready_file": path to the ML-ready CSV (written next to the input)
          "feature_engineering_log": FeatureEngineeringLog JSON with per-action
            results, columns added/removed totals, and success counts.
    """
    import pandas as pd
    from src.agents.feature_transformer_agent import execute_feature_engineering
    from src.models.schemas import FeatureEngineeringPlan

    df = pd.read_csv(cleaned_path)
    plan = FeatureEngineeringPlan.model_validate_json(fe_plan_json)

    ml_df, fe_log = await execute_feature_engineering(df, plan)

    # Write ML-ready file alongside the cleaned file
    base, _ = os.path.splitext(cleaned_path)
    ml_path = base + "_ml_ready.csv"
    ml_df.to_csv(ml_path, index=False)

    return json.dumps({
        "ml_ready_file": ml_path,
        "feature_engineering_log": json.loads(fe_log.model_dump_json()),
    }, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
