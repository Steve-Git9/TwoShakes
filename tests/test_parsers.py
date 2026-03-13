"""Parser tests — verifies all 4 parsers produce non-empty DataFrames."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import logging
logging.basicConfig(level=logging.WARNING)

from src.parsers.csv_parser import parse_csv
from src.parsers.excel_parser import parse_excel
from src.parsers.json_parser import parse_json
from src.parsers.xml_parser import parse_xml

DATA = os.path.join(os.path.dirname(__file__), "..", "test_data")

def test_csv():
    df, meta = parse_csv(os.path.normpath(f"{DATA}/messy_sales.csv"))
    assert meta.row_count > 0
    assert meta.col_count >= 12
    assert meta.file_format == "csv"
    print(f"  [PASS] CSV: {meta.row_count} rows x {meta.col_count} cols")

def test_excel():
    df, meta = parse_excel(os.path.normpath(f"{DATA}/messy_employees.xlsx"))
    assert meta.row_count > 0
    assert meta.col_count >= 5
    assert meta.file_format == "xlsx"
    assert meta.sheet_names is not None and len(meta.sheet_names) >= 2
    print(f"  [PASS] Excel: {meta.row_count} rows x {meta.col_count} cols, sheets={meta.sheet_names}")

def test_json():
    df, meta = parse_json(os.path.normpath(f"{DATA}/nested_api_response.json"))
    assert meta.row_count > 0
    assert meta.col_count >= 5, f"Expected >=5 cols, got {meta.col_count}"
    assert meta.file_format == "json"
    # Nested fields should be flattened
    assert any("address" in c or "." in c for c in df.columns), f"Nested fields not flattened: {df.columns.tolist()}"
    print(f"  [PASS] JSON: {meta.row_count} rows x {meta.col_count} cols, cols={df.columns.tolist()[:5]}...")

def test_xml():
    df, meta = parse_xml(os.path.normpath(f"{DATA}/messy_products.xml"))
    assert meta.row_count > 0
    assert meta.col_count >= 4
    assert meta.file_format == "xml"
    print(f"  [PASS] XML: {meta.row_count} rows x {meta.col_count} cols, cols={df.columns.tolist()}")

def test_dispatcher():
    from src.parsers import parse_file
    for fname, expected_fmt in [
        ("messy_sales.csv", "csv"),
        ("messy_employees.xlsx", "xlsx"),
        ("nested_api_response.json", "json"),
        ("messy_products.xml", "xml"),
    ]:
        df, meta = parse_file(os.path.normpath(f"{DATA}/{fname}"))
        assert meta.row_count > 0, f"{fname}: no rows"
        assert meta.file_format == expected_fmt, f"{fname}: expected {expected_fmt}, got {meta.file_format}"
        print(f"  [PASS] Dispatcher {fname}: {meta.row_count} rows x {meta.col_count} cols")

if __name__ == "__main__":
    print("\n=== Parser Tests ===")
    test_csv()
    test_excel()
    test_json()
    test_xml()
    test_dispatcher()
    print("\nAll parser tests passed.")
