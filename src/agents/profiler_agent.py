"""
Profiler Agent — Stage 1: statistical profiling (pure Python/pandas).
                Stage 2: semantic analysis via LLM (AgentClient).
"""

import json
import logging
import random
from datetime import datetime

import numpy as np
import pandas as pd

from src.agents import AgentClient
from src.models.schemas import (
    ColumnProfile,
    ColumnType,
    DuplicateInfo,
    FileMetadata,
    MissingValueInfo,
    ProfileReport,
)

logger = logging.getLogger(__name__)

# Values that represent "missing" beyond plain NaN
MISSING_PATTERNS = {
    "N/A", "NA", "n/a", "na", "null", "NULL", "Null",
    "none", "None", "NONE", "", "-", ".", "999", "-999",
}

BOOLEAN_VALUES = {"true", "false", "yes", "no", "1", "0", "t", "f", "y", "n"}

_PROFILER_INSTRUCTIONS = """\
You are a data profiling expert. You receive statistical analysis results and sample data.
Your job is to:
1. Infer the semantic meaning of each column from its name and values \
(e.g., "amt" → currency, "dob" → date of birth, "ph" → phone number).
2. Identify groups of categorical values that likely represent the same entity \
(e.g., "US", "USA", "United States" all mean the same country).
3. Spot any data quality issues that pure statistics might miss.
4. Generate an overall quality summary in plain English.

Respond ONLY in valid JSON matching this exact schema:
{
  "column_insights": [
    {
      "column_name": "string",
      "inferred_semantic_type": "string or null",
      "inconsistencies": ["list of strings describing inconsistent values"],
      "additional_issues": ["list of strings"]
    }
  ],
  "overall_summary": "A 2-3 sentence summary of the dataset quality",
  "key_issues": ["top 5 most important issues to fix"],
  "quality_score": 0
}
"""


# ---------------------------------------------------------------------------
# Stage 1 — statistical helpers
# ---------------------------------------------------------------------------

def _is_missing(val: str) -> bool:
    if pd.isna(val):
        return True
    return str(val).strip() in MISSING_PATTERNS


def _count_missing(series: pd.Series) -> tuple[int, list[str]]:
    """Return (count, list_of_encoded_forms)."""
    encoded: set[str] = set()
    count = 0
    for v in series:
        if _is_missing(v):
            count += 1
            encoded.add(str(v).strip() if not pd.isna(v) else "")
    return count, sorted(encoded)


def _try_numeric(series: pd.Series) -> bool:
    """Try converting series (after stripping currency chars) to float."""
    cleaned = series.dropna().str.strip().str.replace(r"[$,\s]", "", regex=True)
    try:
        pd.to_numeric(cleaned, errors="raise")
        return True
    except Exception:
        return False


def _try_datetime(series: pd.Series) -> bool:
    from dateutil.parser import parse, ParserError
    non_null = series.dropna().head(10)
    if len(non_null) == 0:
        return False
    successes = 0
    for v in non_null:
        try:
            parse(str(v), fuzzy=False)
            successes += 1
        except (ParserError, ValueError, OverflowError):
            pass
    return successes / len(non_null) >= 0.7


def _try_boolean(series: pd.Series) -> bool:
    vals = set(series.dropna().str.strip().str.lower().unique())
    return len(vals) > 0 and vals.issubset(BOOLEAN_VALUES)


def _detect_type(series: pd.Series) -> ColumnType:
    non_missing = series[~series.apply(_is_missing)].dropna()
    if len(non_missing) == 0:
        return ColumnType.UNKNOWN

    if _try_boolean(non_missing):
        return ColumnType.BOOLEAN
    if _try_numeric(non_missing):
        return ColumnType.NUMERIC
    if _try_datetime(non_missing):
        return ColumnType.DATETIME

    cardinality = non_missing.nunique() / max(len(non_missing), 1)
    if cardinality < 0.2:
        return ColumnType.CATEGORICAL
    return ColumnType.TEXT


def _numeric_stats(series: pd.Series) -> dict:
    cleaned = pd.to_numeric(
        series.str.strip().str.replace(r"[$,\s]", "", regex=True),
        errors="coerce",
    )
    q1, q3 = cleaned.quantile(0.25), cleaned.quantile(0.75)
    iqr = q3 - q1
    outliers = int(((cleaned < q1 - 1.5 * iqr) | (cleaned > q3 + 1.5 * iqr)).sum())
    return {
        "mean": round(float(cleaned.mean()), 4),
        "median": round(float(cleaned.median()), 4),
        "std": round(float(cleaned.std()), 4),
        "min": float(cleaned.min()),
        "max": float(cleaned.max()),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "outlier_count": outliers,
        "non_numeric_count": int(pd.to_numeric(series.str.strip().str.replace(r"[$,\s]", "", regex=True), errors="coerce").isna().sum() - series.isna().sum()),
    }


def _quality_score(missing_pct: float, col_type: ColumnType, stats: dict | None) -> float:
    score = 100.0
    score -= missing_pct * 1.5
    if col_type == ColumnType.UNKNOWN:
        score -= 20
    if stats:
        outlier_pct = stats.get("outlier_count", 0) / max(stats.get("mean", 1) or 1, 1)
        score -= min(outlier_pct * 5, 15)
        if stats.get("non_numeric_count", 0) > 0:
            score -= 10
    return max(0.0, min(100.0, score))


def _profile_column(col: str, series: pd.Series, total_rows: int) -> ColumnProfile:
    missing_count, encoded_as = _count_missing(series)
    missing_pct = (missing_count / total_rows * 100) if total_rows else 0

    non_missing = series[~series.apply(_is_missing)].dropna()
    unique_count = int(non_missing.nunique())
    cardinality_ratio = unique_count / max(len(non_missing), 1)

    # Sample values
    sample_pool = non_missing.dropna().tolist()
    sample_values = [str(v) for v in random.sample(sample_pool, min(5, len(sample_pool)))]

    col_type = _detect_type(series)

    stats = None
    top_values = None
    issues: list[str] = []

    if col_type == ColumnType.NUMERIC:
        stats = _numeric_stats(non_missing)
        if stats["outlier_count"] > 0:
            issues.append(f"{stats['outlier_count']} outlier(s) detected (IQR method)")
        if stats["non_numeric_count"] > 0:
            issues.append(f"{stats['non_numeric_count']} non-numeric value(s) in numeric column")
    elif col_type == ColumnType.CATEGORICAL:
        vc = non_missing.str.strip().value_counts().head(10)
        top_values = vc.to_dict()
        # Check for inconsistent casing / variants
        lower_unique = non_missing.str.strip().str.lower().nunique()
        if lower_unique < unique_count:
            issues.append(
                f"Case inconsistency detected ({unique_count} variants → {lower_unique} unique after normalising)"
            )

    if missing_count > 0:
        issues.append(f"{missing_count} missing values ({missing_pct:.1f}%)")

    quality = _quality_score(missing_pct, col_type, stats)

    return ColumnProfile(
        column_name=col,
        detected_type=col_type,
        missing=MissingValueInfo(
            count=missing_count,
            percentage=round(missing_pct, 2),
            encoded_as=encoded_as,
        ),
        unique_count=unique_count,
        cardinality_ratio=round(cardinality_ratio, 4),
        sample_values=sample_values,
        stats=stats,
        top_values=top_values,
        issues=issues,
        quality_score=round(quality, 2),
    )


def _detect_duplicates(df: pd.DataFrame) -> DuplicateInfo:
    # Exact duplicates
    dup_mask = df.duplicated(keep=False)
    exact_dup_count = int(dup_mask.sum())
    # Group indices
    if exact_dup_count > 0:
        dup_groups: dict[tuple, list[int]] = {}
        for idx, row in df[dup_mask].iterrows():
            key = tuple(row.values)
            dup_groups.setdefault(key, []).append(int(idx))
        exact_dup_row_indices = list(dup_groups.values())
    else:
        exact_dup_row_indices = []

    # Fuzzy duplicates (string columns concatenated, thefuzz ratio)
    fuzzy_groups: list[dict] = []
    try:
        from thefuzz import fuzz  # type: ignore

        str_cols = df.select_dtypes(include="object").columns.tolist()
        if str_cols:
            concat = df[str_cols].fillna("").apply(lambda r: " ".join(r.values), axis=1)
            visited: set[int] = set()
            for i in range(len(concat)):
                if i in visited:
                    continue
                group_indices = [i]
                for j in range(i + 1, len(concat)):
                    if j in visited:
                        continue
                    ratio = fuzz.ratio(concat.iloc[i], concat.iloc[j])
                    if ratio >= 90:
                        group_indices.append(j)
                        visited.add(j)
                if len(group_indices) > 1:
                    visited.add(i)
                    fuzzy_groups.append({"indices": group_indices, "similarity": ratio})
    except ImportError:
        logger.warning("thefuzz not installed — skipping fuzzy duplicate detection")

    return DuplicateInfo(
        exact_duplicate_count=exact_dup_count,
        exact_duplicate_row_indices=exact_dup_row_indices,
        fuzzy_duplicate_count=sum(len(g["indices"]) for g in fuzzy_groups),
        fuzzy_duplicate_groups=fuzzy_groups,
    )


# ---------------------------------------------------------------------------
# Stage 2 — LLM semantic analysis
# ---------------------------------------------------------------------------

def _build_llm_payload(df: pd.DataFrame, column_profiles: list[ColumnProfile]) -> str:
    columns_info = []
    for cp in column_profiles:
        columns_info.append({
            "column_name": cp.column_name,
            "detected_type": cp.detected_type.value,
            "missing_pct": cp.missing.percentage,
            "unique_count": cp.unique_count,
            "sample_values": cp.sample_values,
            "issues": cp.issues,
            "top_values": cp.top_values,
        })
    payload = {
        "dataset_shape": {"rows": len(df), "cols": len(df.columns)},
        "columns": columns_info,
    }
    return json.dumps(payload, default=str)


async def _llm_semantic_analysis(df: pd.DataFrame, column_profiles: list[ColumnProfile]) -> dict:
    client = AgentClient(
        name="ProfilerAgent",
        instructions=_PROFILER_INSTRUCTIONS,
        json_mode=True,
    )
    payload = _build_llm_payload(df, column_profiles)
    raw = await client.run(payload)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"LLM returned invalid JSON: {e}\nRaw: {raw[:500]}")
        return {
            "column_insights": [],
            "overall_summary": "LLM analysis unavailable.",
            "key_issues": [],
            "quality_score": 50,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def profile_dataframe(df: pd.DataFrame, metadata: FileMetadata) -> ProfileReport:
    """
    Full profiling pipeline:
    Stage 1 — statistical profiling (no LLM).
    Stage 2 — LLM semantic enrichment.
    Returns a ProfileReport.
    """
    logger.info(f"Stage 1: statistical profiling of {len(df.columns)} columns")
    total_rows = len(df)
    column_profiles: list[ColumnProfile] = []

    for col in df.columns:
        cp = _profile_column(col, df[col].astype(str), total_rows)
        column_profiles.append(cp)

    duplicates = _detect_duplicates(df)

    logger.info("Stage 2: LLM semantic analysis")
    llm_result = await _llm_semantic_analysis(df, column_profiles)

    # Merge LLM insights into column profiles
    insights_by_col: dict[str, dict] = {
        ins["column_name"]: ins
        for ins in llm_result.get("column_insights", [])
    }
    enriched_profiles: list[ColumnProfile] = []
    for cp in column_profiles:
        ins = insights_by_col.get(cp.column_name, {})
        extra_issues = ins.get("inconsistencies", []) + ins.get("additional_issues", [])
        enriched = cp.model_copy(update={
            "inferred_semantic_type": ins.get("inferred_semantic_type"),
            "issues": cp.issues + [i for i in extra_issues if i not in cp.issues],
        })
        enriched_profiles.append(enriched)

    overall_quality = float(llm_result.get("quality_score", 0)) or (
        sum(cp.quality_score for cp in enriched_profiles) / max(len(enriched_profiles), 1)
    )

    return ProfileReport(
        file_metadata=metadata,
        columns=enriched_profiles,
        duplicates=duplicates,
        overall_quality_score=round(overall_quality, 2),
        summary=llm_result.get("overall_summary", ""),
        key_issues=llm_result.get("key_issues", []),
        generated_at=datetime.utcnow(),
    )
