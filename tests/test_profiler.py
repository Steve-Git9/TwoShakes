"""
Test script: parse messy_sales.csv → run Profiler Agent → print ProfileReport.

Run with:
    python tests/test_profiler.py
"""

import asyncio
import json
import logging
import sys
import os

# Force UTF-8 stdout on Windows so Unicode characters from LLM don't crash print()
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

from src.parsers.csv_parser import parse_csv
from src.agents.profiler_agent import profile_dataframe


async def main():
    csv_path = os.path.join(os.path.dirname(__file__), "..", "test_data", "messy_sales.csv")
    csv_path = os.path.normpath(csv_path)

    print(f"\n{'='*60}")
    print("Step 1: Parsing CSV")
    print(f"{'='*60}")
    df, metadata = parse_csv(csv_path)
    print(f"  Rows: {metadata.row_count}")
    print(f"  Columns: {metadata.col_count}")
    print(f"  Encoding: {metadata.encoding}")
    if metadata.parse_warnings:
        print(f"  Warnings: {metadata.parse_warnings}")

    print(f"\n{'='*60}")
    print("Step 2: Running Profiler Agent (statistical + LLM)")
    print(f"{'='*60}")
    report = await profile_dataframe(df, metadata)

    print(f"\n{'='*60}")
    print("ProfileReport (JSON)")
    print(f"{'='*60}")
    print(report.model_dump_json(indent=2))

    # --- Assertions ---
    print(f"\n{'='*60}")
    print("Assertions")
    print(f"{'='*60}")

    assert report.file_metadata.row_count > 0, "Expected row_count > 0"
    print("  [PASS] row_count > 0")

    assert len(report.columns) > 0, "Expected at least one column profile"
    print(f"  [PASS] {len(report.columns)} columns profiled")

    assert 0 <= report.overall_quality_score <= 100, (
        f"quality_score out of range: {report.overall_quality_score}"
    )
    print(f"  [PASS] overall_quality_score = {report.overall_quality_score}")

    all_issues = [issue for col in report.columns for issue in col.issues] + report.key_issues
    assert len(all_issues) >= 1, "Expected at least 1 issue detected"
    print(f"  [PASS] {len(all_issues)} issue(s) detected across all columns")

    print(f"\n{'='*60}")
    print("All assertions passed. Pipeline is working.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
