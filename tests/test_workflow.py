"""
End-to-end pipeline test:
  Parse messy_sales.csv -> Profile -> Strategy -> Clean (auto-approve all) -> Validate
  Prints before/after quality scores and saves cleaned data to test_data/messy_sales_cleaned.csv

Run with:
    python tests/test_workflow.py
"""

import asyncio
import sys
import os

# Force UTF-8 stdout on Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

from src.parsers.csv_parser import parse_csv
from src.agents.profiler_agent import profile_dataframe
from src.agents.strategy_agent import generate_cleaning_plan
from src.agents.cleaner_agent import execute_cleaning_plan
from src.agents.validator_agent import validate


def _hr(label: str = ""):
    print(f"\n{'='*60}")
    if label:
        print(label)
        print(f"{'='*60}")


async def main():
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    csv_path = os.path.join(root, "test_data", "messy_sales.csv")
    out_path = os.path.join(root, "test_data", "messy_sales_cleaned.csv")

    # Step 1: Parse
    _hr("Step 1: Parsing messy_sales.csv")
    df, metadata = parse_csv(csv_path)
    print(f"  Rows: {metadata.row_count}  Columns: {metadata.col_count}  Encoding: {metadata.encoding}")

    # Step 2: Profile
    _hr("Step 2: Profiling")
    profile = await profile_dataframe(df, metadata)
    print(f"  Overall quality score (before): {profile.overall_quality_score}")
    print(f"  Key issues:")
    for issue in profile.key_issues:
        print(f"    - {issue}")
    print(f"  Exact duplicates: {profile.duplicates.exact_duplicate_count}")

    # Step 3: Generate Cleaning Plan
    _hr("Step 3: Generating Cleaning Plan")
    plan = await generate_cleaning_plan(profile, df)
    print(f"  {len(plan.actions)} actions proposed")
    print(f"  Estimated rows affected: {plan.estimated_rows_affected}")
    for action in plan.actions:
        flag = "[HIGH]" if action.priority.value == "high" else f"[{action.priority.value.upper()}]"
        print(f"    {flag} {action.action_type.value} on '{action.column_name or 'all rows'}' -- {action.description}")

    # Step 4: Execute (auto-approve all)
    _hr("Step 4: Executing Cleaning Plan (all approved)")
    for action in plan.actions:
        action.approved = True

    df_clean, log = await execute_cleaning_plan(df.copy(), plan)

    print(f"  Actions executed: {log.total_actions_executed}")
    print(f"  Actions succeeded: {log.total_actions_succeeded}")
    print(f"  Total rows modified: {log.total_rows_modified}")
    print(f"  Rows before: {len(df)}  |  Rows after: {len(df_clean)}")

    # Step 5: Validate
    _hr("Step 5: Validation")
    report = await validate(df, df_clean, profile, log)

    print(f"\n  Quality score:  {report.before_quality_score:.1f} -> {report.after_quality_score:.1f}  "
          f"(improvement: {report.improvement_percentage:+.1f})")
    print(f"\n  Certificate:\n  {report.data_quality_certificate}")
    print(f"\n  Validation checks:")
    for check in report.checks:
        icon = "OK" if check.passed else "!!"
        print(f"    [{icon}] {check.check_name}: {check.details}")

    # Step 6: Save cleaned CSV
    _hr("Step 6: Saving cleaned data")
    df_clean.to_csv(out_path, index=False)
    print(f"  Saved to: {out_path}")
    print(f"  Shape: {df_clean.shape}")

    # Assertions
    _hr("Assertions")
    assert report.row_count_before > 0, "Expected rows before > 0"
    print("  [PASS] row_count_before > 0")

    assert report.row_count_after > 0, "Expected rows after > 0"
    print("  [PASS] row_count_after > 0")

    assert 0 <= report.after_quality_score <= 100, f"Score out of range: {report.after_quality_score}"
    print(f"  [PASS] after_quality_score = {report.after_quality_score}")

    assert log.total_actions_executed > 0, "Expected at least 1 action executed"
    print(f"  [PASS] {log.total_actions_executed} actions executed")

    assert os.path.exists(out_path), "Cleaned CSV not saved"
    print(f"  [PASS] cleaned CSV saved to {out_path}")

    _hr()
    print("All assertions passed. Full pipeline is working.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
