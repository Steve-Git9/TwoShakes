"""
Feature Engineering Agent
===========================
Analyzes a CLEANED DataFrame and recommends feature engineering
transformations to make the data ML-ready.

Two-phase approach:
  1. Statistical analysis (Python / pandas — no LLM)
  2. LLM recommendation (via AgentClient) using the stats as input

Returns a FeatureEngineeringPlan for human review.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.agents import AgentClient
from src.models.schemas import (
    FeatureEngineeringAction,
    FeatureEngineeringActionType,
    FeatureEngineeringPlan,
    Priority,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a feature engineering expert preparing data for machine learning.
You receive a statistical analysis of a cleaned dataset. Your job is to recommend
the optimal feature engineering transformations to make this data ML-ready.

AVAILABLE TRANSFORMATIONS AND WHEN TO RECOMMEND EACH:

ENCODING (for categorical columns):
- one_hot_encode: Use for NOMINAL categorical columns with LOW cardinality (2-10 unique values). Creates binary columns for each category. Parameters: {"drop_first": true} to avoid multicollinearity.
  WARNING: Do NOT use if cardinality > 15 — causes dimensionality explosion.
- label_encode: Use for ORDINAL categorical columns where order matters (e.g., "low" < "medium" < "high"). Parameters: {"order": ["low", "medium", "high"]}
- ordinal_encode: Same as label_encode but when you want to specify a custom mapping. Parameters: {"mapping": {"low": 0, "medium": 1, "high": 2}}
- target_encode: Use for HIGH cardinality categorical columns (>10 unique values) when a target column exists. Replaces categories with the mean of the target variable. Parameters: {"target_column": "price", "smoothing": 1.0}
- frequency_encode: Use for HIGH cardinality categorical columns when no target exists. Replaces each category with its frequency count. Parameters: {}

SCALING (for numeric columns):
- standard_scale: Use when data is approximately normally distributed. Centers to mean=0, std=1. Best for: linear regression, logistic regression, SVM, neural networks. Parameters: {}
- min_max_scale: Use when you need values in [0, 1] range. Best for: neural networks, KNN, algorithms sensitive to magnitude. Parameters: {"feature_range": [0, 1]}
- robust_scale: Use when data has OUTLIERS. Uses median and IQR instead of mean and std. Best for: data with outliers that you want to keep. Parameters: {}
- max_abs_scale: Use for SPARSE data. Scales by maximum absolute value to [-1, 1]. Parameters: {}

DISTRIBUTION TRANSFORMS (for skewed numeric columns):
- log_transform: Use for RIGHT-SKEWED data (skewness > 1.0) with all POSITIVE values. Compresses large values. Parameters: {"base": "natural"} or {"base": 10}
  WARNING: Fails if column has zero or negative values. Suggest adding offset if needed.
- power_transform: Use for skewed data (any direction). Yeo-Johnson works with negative values, Box-Cox requires positive only. Parameters: {"method": "yeo-johnson"} or {"method": "box-cox"}
- quantile_transform: Use to force any distribution into a normal or uniform shape. Aggressive — use when other transforms fail. Parameters: {"output_distribution": "normal"} or {"output_distribution": "uniform"}

DISCRETIZATION:
- binning: Use to convert continuous numeric into categorical bins. Parameters: {"n_bins": 5, "strategy": "quantile"} or {"strategy": "uniform"} or {"strategy": "kmeans"}

FEATURE CREATION:
- interaction_features: Use when pairs of features might have multiplicative effects. Parameters: {"column_pairs": [["width", "height"]]}
- polynomial_features: Use to capture nonlinear relationships. Parameters: {"degree": 2, "interaction_only": false, "columns": ["col1", "col2"]}
  WARNING: Can cause dimensionality explosion. Only use on 2-5 selected columns.

FEATURE SELECTION / REDUCTION:
- drop_low_variance: Remove columns with near-zero variance. Parameters: {"threshold": 0.01}
- drop_high_cardinality: Remove categorical columns with too many unique values. Parameters: {"max_cardinality": 50}
- drop_highly_correlated: Remove one of each pair of highly correlated columns. Parameters: {"threshold": 0.95}

RULES:
1. Recommend encoding for ALL categorical columns.
2. Recommend scaling for ALL numeric columns — choose the right scaler.
3. Only recommend distribution transforms for columns with |skewness| > 1.0.
4. Only recommend feature creation if it makes domain sense.
5. Only recommend feature selection if there are clear candidates.
6. Order: encoding first, then distribution transforms, then scaling, then feature creation, then feature selection.
7. Include "warning" field when an action has caveats.

Respond ONLY in valid JSON matching this exact schema:
{
  "ml_task_hint": "classification" | "regression" | "clustering" | "general",
  "actions": [
    {
      "id": "fe_001",
      "column_name": "column_name" or null,
      "columns": ["col1", "col2"] or null,
      "action_type": "one_hot_encode",
      "description": "Human-readable description",
      "parameters": {},
      "reason": "Why this is recommended",
      "impact": "What this changes about the data",
      "priority": "high" | "medium" | "low",
      "warning": "optional caveat or null"
    }
  ]
}"""


def _analyze_dataframe(df: pd.DataFrame, target_column: Optional[str]) -> dict:
    """Phase 1: Pure Python statistical analysis. Returns a dict for the LLM prompt."""
    analysis: dict = {
        "shape": {"rows": len(df), "columns": len(df.columns)},
        "target_column": target_column,
        "columns": [],
        "high_correlation_pairs": [],
        "low_variance_columns": [],
    }

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    for col in df.columns:
        col_info: dict = {
            "name": col,
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique()),
            "cardinality_ratio": round(df[col].nunique() / max(len(df), 1), 4),
        }

        if col in numeric_cols:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            col_info["type"] = "numeric"
            col_info["min"] = float(vals.min()) if len(vals) else None
            col_info["max"] = float(vals.max()) if len(vals) else None
            col_info["mean"] = round(float(vals.mean()), 4) if len(vals) else None
            col_info["std"] = round(float(vals.std()), 4) if len(vals) else None
            col_info["skewness"] = round(float(vals.skew()), 4) if len(vals) > 2 else 0.0
            col_info["kurtosis"] = round(float(vals.kurt()), 4) if len(vals) > 3 else 0.0
            col_info["has_negatives"] = bool((vals < 0).any())
            col_info["has_zeros"] = bool((vals == 0).any())
            # Variance check
            if len(vals) > 1 and vals.std() > 0:
                cv = abs(vals.std() / vals.mean()) if vals.mean() != 0 else float("inf")
                col_info["coefficient_of_variation"] = round(cv, 4)
            else:
                col_info["coefficient_of_variation"] = 0.0
            if target_column and target_column in df.columns and col != target_column:
                try:
                    corr_with_target = round(
                        float(df[col].corr(df[target_column])), 4
                    )
                    col_info["correlation_with_target"] = corr_with_target
                except Exception:
                    pass
        elif col in cat_cols:
            col_info["type"] = "categorical"
            col_info["top_values"] = df[col].value_counts().head(5).to_dict()
        else:
            col_info["type"] = "other"

        analysis["columns"].append(col_info)

    # High-correlation pairs among numeric columns
    if len(numeric_cols) >= 2:
        try:
            corr_matrix = df[numeric_cols].corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            for col in upper.columns:
                for idx in upper.index:
                    val = upper.loc[idx, col]
                    if pd.notna(val) and val > 0.90:
                        analysis["high_correlation_pairs"].append({
                            "col1": idx,
                            "col2": col,
                            "correlation": round(float(val), 4),
                        })
        except Exception:
            pass

    # Low-variance columns
    for col in numeric_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(vals) > 1 and vals.std() / (abs(vals.mean()) + 1e-9) < 0.01:
            analysis["low_variance_columns"].append(col)

    return analysis


async def recommend_feature_engineering(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> FeatureEngineeringPlan:
    """
    Analyze a cleaned DataFrame and return a FeatureEngineeringPlan.

    Args:
        df:             Cleaned DataFrame (post-validation).
        target_column:  Optional name of the prediction target column.

    Returns:
        FeatureEngineeringPlan with recommended actions (all pre-approved).
    """
    logger.info(
        f"[FeatureEngineeringAgent] Analyzing {len(df)} rows × {len(df.columns)} cols"
        + (f" | target={target_column}" if target_column else "")
    )

    # Phase 1: Statistical analysis
    stats = _analyze_dataframe(df, target_column)
    logger.info(
        f"[FeatureEngineeringAgent] Stats: "
        f"{len([c for c in stats['columns'] if c.get('type')=='numeric'])} numeric, "
        f"{len([c for c in stats['columns'] if c.get('type')=='categorical'])} categorical, "
        f"{len(stats['high_correlation_pairs'])} high-corr pairs"
    )

    # Phase 2: LLM recommendation
    client = AgentClient(
        name="FeatureEngineeringAgent",
        instructions=_SYSTEM_PROMPT,
        json_mode=True,
    )

    user_msg = f"Dataset analysis:\n{json.dumps(stats, indent=2, default=str)}"
    if target_column:
        user_msg += f"\n\nTarget column for prediction: {target_column}"

    raw = await client.run(user_msg)

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error(f"[FeatureEngineeringAgent] JSON parse failed: {e}\nRaw: {raw[:500]}")
        # Return empty plan rather than crashing
        return FeatureEngineeringPlan(
            target_column=target_column,
            ml_task_hint="general",
            actions=[],
        )

    actions: list[FeatureEngineeringAction] = []
    for i, a in enumerate(data.get("actions", [])):
        try:
            action = FeatureEngineeringAction(
                id=a.get("id", f"fe_{i+1:03d}"),
                column_name=a.get("column_name"),
                columns=a.get("columns"),
                action_type=FeatureEngineeringActionType(a["action_type"]),
                description=a.get("description", ""),
                parameters=a.get("parameters") or {},
                reason=a.get("reason", ""),
                impact=a.get("impact", ""),
                priority=Priority(a.get("priority", "medium")),
                approved=True,
                warning=a.get("warning"),
            )
            actions.append(action)
        except Exception as e:
            logger.warning(f"[FeatureEngineeringAgent] Skipping malformed action {a}: {e}")

    plan = FeatureEngineeringPlan(
        target_column=target_column,
        ml_task_hint=data.get("ml_task_hint", "general"),
        actions=actions,
    )
    logger.info(
        f"[FeatureEngineeringAgent] Plan: {len(actions)} actions, "
        f"task_hint={plan.ml_task_hint}"
    )
    return plan
