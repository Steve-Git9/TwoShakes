"""
Tests for the Feature Engineering pipeline.

Runs without Azure credentials — uses the transformation functions directly
and mocks the LLM call for the agent test.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest

from src.models.schemas import (
    FeatureEngineeringAction,
    FeatureEngineeringActionType,
    FeatureEngineeringPlan,
    Priority,
)
from src.transformations.feature_engineering import (
    apply_feature_engineering_action,
    binning,
    drop_high_cardinality,
    drop_highly_correlated,
    drop_low_variance,
    frequency_encode,
    interaction_features,
    label_encode,
    log_transform,
    min_max_scale,
    one_hot_encode,
    polynomial_features,
    power_transform,
    robust_scale,
    standard_scale,
)
from src.agents.feature_transformer_agent import execute_feature_engineering


# ─────────────────────────────────────────────────────────────────────────────
# Sample DataFrames
# ─────────────────────────────────────────────────────────────────────────────

def make_df() -> pd.DataFrame:
    """A small DataFrame that covers all column types."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "age": np.random.randint(18, 70, n).astype(float),
        "salary": np.random.lognormal(10, 1, n),   # right-skewed
        "score": np.random.uniform(0, 100, n),
        "category": np.random.choice(["A", "B", "C"], n),
        "city": np.random.choice([f"City_{i}" for i in range(60)], n),  # high cardinality
        "tier": np.random.choice(["low", "medium", "high"], n),
        "constant": 5.0,                             # zero variance
    })


# ─────────────────────────────────────────────────────────────────────────────
# Encoding tests
# ─────────────────────────────────────────────────────────────────────────────

def test_one_hot_encode():
    df = make_df()
    result, rec = one_hot_encode(df, "category", drop_first=True)
    assert "category" not in result.columns
    assert any(c.startswith("category_") for c in result.columns)
    assert rec.success
    assert len(rec.columns_added) >= 1
    assert "category" in rec.columns_removed


def test_label_encode_with_order():
    df = make_df()
    result, rec = label_encode(df, "tier", order=["low", "medium", "high"])
    assert result["tier"].dtype in [int, "int64", "int32"]
    assert result["tier"].min() >= 0
    assert rec.success


def test_frequency_encode():
    df = make_df()
    result, rec = frequency_encode(df, "category")
    assert pd.api.types.is_numeric_dtype(result["category"])
    assert rec.success


# ─────────────────────────────────────────────────────────────────────────────
# Scaling tests
# ─────────────────────────────────────────────────────────────────────────────

def test_standard_scale():
    df = make_df()
    result, rec = standard_scale(df, "age")
    assert abs(result["age"].mean()) < 0.01
    assert abs(result["age"].std() - 1.0) < 0.1
    assert rec.success


def test_min_max_scale():
    df = make_df()
    result, rec = min_max_scale(df, "score")
    assert result["score"].min() >= -0.001
    assert result["score"].max() <= 1.001
    assert rec.success


def test_robust_scale():
    df = make_df()
    result, rec = robust_scale(df, "salary")
    assert rec.success
    assert pd.api.types.is_numeric_dtype(result["salary"])


# ─────────────────────────────────────────────────────────────────────────────
# Distribution transform tests
# ─────────────────────────────────────────────────────────────────────────────

def test_log_transform_positive_data():
    df = make_df()
    result, rec = log_transform(df, "salary")
    assert rec.success
    assert not result["salary"].isna().any()


def test_log_transform_with_zeros():
    df = pd.DataFrame({"val": [0, 1, 2, 3, 100]})
    result, rec = log_transform(df, "val")
    assert rec.success
    assert not result["val"].isna().any()


def test_power_transform():
    df = make_df()
    result, rec = power_transform(df, "salary", method="yeo-johnson")
    assert rec.success
    assert not result["salary"].isna().any()


# ─────────────────────────────────────────────────────────────────────────────
# Discretization tests
# ─────────────────────────────────────────────────────────────────────────────

def test_binning():
    df = make_df()
    result, rec = binning(df, "age", n_bins=5, strategy="quantile")
    assert "age_binned" in result.columns
    assert result["age_binned"].nunique() <= 5
    assert rec.success


# ─────────────────────────────────────────────────────────────────────────────
# Feature creation tests
# ─────────────────────────────────────────────────────────────────────────────

def test_interaction_features():
    df = make_df()
    result, rec = interaction_features(df, [["age", "score"]])
    assert "age_x_score" in result.columns
    expected = df["age"] * df["score"]
    pd.testing.assert_series_equal(result["age_x_score"], expected, check_names=False)
    assert rec.success


def test_polynomial_features():
    df = make_df()
    result, rec = polynomial_features(df, ["age", "score"], degree=2)
    assert len(result.columns) > len(df.columns)
    assert rec.success
    assert len(rec.columns_added) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Feature selection tests
# ─────────────────────────────────────────────────────────────────────────────

def test_drop_low_variance():
    df = make_df()
    result, rec = drop_low_variance(df, threshold=0.01)
    assert "constant" not in result.columns
    assert rec.success
    assert "constant" in rec.columns_removed


def test_drop_high_cardinality():
    df = make_df()
    result, rec = drop_high_cardinality(df, max_cardinality=49)
    assert "city" not in result.columns
    assert rec.success


def test_drop_highly_correlated():
    # Two identical columns should trigger removal
    df = pd.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "b": [1.0, 2.0, 3.0, 4.0, 5.0],  # perfectly correlated with a
        "c": [5.0, 4.0, 3.0, 2.0, 1.0],
    })
    result, rec = drop_highly_correlated(df, threshold=0.95)
    # One of a/b should be dropped
    assert len(result.columns) < len(df.columns)
    assert rec.success


# ─────────────────────────────────────────────────────────────────────────────
# Dispatcher test
# ─────────────────────────────────────────────────────────────────────────────

def test_apply_feature_engineering_action_dispatcher():
    df = make_df()
    action = FeatureEngineeringAction(
        id="test_001",
        column_name="category",
        action_type=FeatureEngineeringActionType.ONE_HOT_ENCODE,
        description="OHE category",
        parameters={"drop_first": True},
        reason="categorical",
        impact="creates binary cols",
        priority=Priority.HIGH,
    )
    result, rec = apply_feature_engineering_action(df, action)
    assert "category" not in result.columns
    assert rec.success


# ─────────────────────────────────────────────────────────────────────────────
# Full executor test (no LLM)
# ─────────────────────────────────────────────────────────────────────────────

def test_execute_feature_engineering_no_llm():
    """Build a hand-crafted plan and execute it end-to-end."""
    df = make_df()

    plan = FeatureEngineeringPlan(
        ml_task_hint="regression",
        actions=[
            FeatureEngineeringAction(
                id="fe_001",
                column_name="category",
                action_type=FeatureEngineeringActionType.ONE_HOT_ENCODE,
                description="Encode category",
                parameters={"drop_first": True},
                reason="categorical → numeric",
                impact="binary cols",
                priority=Priority.HIGH,
                approved=True,
            ),
            FeatureEngineeringAction(
                id="fe_002",
                column_name="tier",
                action_type=FeatureEngineeringActionType.LABEL_ENCODE,
                description="Encode tier",
                parameters={"order": ["low", "medium", "high"]},
                reason="ordinal",
                impact="0/1/2",
                priority=Priority.HIGH,
                approved=True,
            ),
            FeatureEngineeringAction(
                id="fe_003",
                column_name="salary",
                action_type=FeatureEngineeringActionType.LOG_TRANSFORM,
                description="Log salary",
                parameters={"base": "natural"},
                reason="right-skewed",
                impact="reduce skew",
                priority=Priority.MEDIUM,
                approved=True,
            ),
            FeatureEngineeringAction(
                id="fe_004",
                column_name="age",
                action_type=FeatureEngineeringActionType.STANDARD_SCALE,
                description="Scale age",
                parameters={},
                reason="normalize range",
                impact="mean=0 std=1",
                priority=Priority.MEDIUM,
                approved=True,
            ),
            # This action is rejected — should be skipped
            FeatureEngineeringAction(
                id="fe_005",
                column_name="score",
                action_type=FeatureEngineeringActionType.MIN_MAX_SCALE,
                description="Scale score",
                parameters={},
                reason="normalize",
                impact="0-1",
                priority=Priority.LOW,
                approved=False,
            ),
        ],
    )

    ml_df, fe_log = asyncio.run(execute_feature_engineering(df, plan))

    # 4 approved actions → 4 executed
    assert fe_log.total_actions_executed == 4
    assert fe_log.total_actions_succeeded == 4

    # Categorical columns encoded → no object dtypes for category/tier
    assert "category" not in ml_df.columns
    assert pd.api.types.is_numeric_dtype(ml_df["tier"])

    # Log transform applied
    assert not ml_df["salary"].isna().any()

    # Standard scale applied to age
    assert abs(ml_df["age"].mean()) < 0.1

    # score not touched (rejected)
    pd.testing.assert_series_equal(ml_df["score"], df["score"])

    print("\n=== FeatureEngineeringLog ===")
    for r in fe_log.records:
        status = "✓" if r.success else "✗"
        print(
            f"  {status} {r.action_id} | {r.action_type.value} | "
            f"+{len(r.columns_added)} -{len(r.columns_removed)} cols"
        )
    print(
        f"\nFinal shape: {len(ml_df)} rows × {len(ml_df.columns)} cols  "
        f"(was {len(df)} × {len(df.columns)})"
    )
    print(f"No NaN values: {not ml_df.isna().any().any()}")
    print(f"All object cols encoded: {list(ml_df.select_dtypes('object').columns) == []}")


if __name__ == "__main__":
    # Run the integration test directly
    test_execute_feature_engineering_no_llm()
    print("\nAll tests would pass (run with: pytest tests/test_feature_engineering.py -v)")
