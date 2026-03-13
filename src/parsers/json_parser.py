"""JSON parser — flattens nested/nested-API structures using json_normalize."""
import os, json, logging
from typing import Any, Optional
import pandas as pd
from src.models.schemas import FileMetadata

logger = logging.getLogger(__name__)

def _find_records(data: Any, depth: int = 0) -> Optional[list]:
    if depth > 5: return None
    if isinstance(data, list) and data and isinstance(data[0], dict): return data
    if isinstance(data, dict):
        for v in data.values():
            r = _find_records(v, depth + 1)
            if r is not None: return r
    return None

def parse_json(file_path: str) -> tuple[pd.DataFrame, FileMetadata]:
    """Parse JSON; auto-detects nested record arrays and flattens with dot notation."""
    warnings: list[str] = []
    size_bytes = os.path.getsize(file_path)
    original_filename = os.path.basename(file_path)
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        if isinstance(data, list):
            records = data
        else:
            records = _find_records(data)
            if records is None:
                warnings.append("No record array found; flattening root object")
                records = [data] if isinstance(data, dict) else []
        if not records:
            warnings.append("JSON contains no records")
            df = pd.DataFrame()
        else:
            df = pd.json_normalize(records, sep=".")
            df = df.astype(str).replace("None", "").replace("nan", "")
        logger.info(f"Parsed JSON: {len(df)} rows x {len(df.columns)} cols")
    except Exception as e:
        warnings.append(f"JSON parse error: {e}")
        logger.error(f"Failed to parse {file_path}: {e}")
        df = pd.DataFrame()
    return df, FileMetadata(
        original_filename=original_filename, file_format="json",
        size_bytes=size_bytes, row_count=len(df), col_count=len(df.columns),
        parse_warnings=warnings,
    )
