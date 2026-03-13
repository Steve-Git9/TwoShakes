"""XML parser — finds the repeating element and normalises to a DataFrame."""
import os, logging
from typing import Any, Optional
import pandas as pd
import xmltodict
from src.models.schemas import FileMetadata

logger = logging.getLogger(__name__)

def _flatten(record: dict, prefix: str = "") -> dict:
    flat: dict = {}
    for key, value in record.items():
        clean = key.lstrip("@").replace("#text", "value")
        full = f"{prefix}.{clean}" if prefix else clean
        if isinstance(value, dict):
            flat.update(_flatten(value, full))
        elif isinstance(value, list):
            flat[full] = "|".join(str(v) for v in value)
        else:
            flat[full] = str(value) if value is not None else ""
    return flat

def _find_list(data: Any, depth: int = 0) -> Optional[list]:
    if depth > 5: return None
    if isinstance(data, list) and data and isinstance(data[0], dict): return data
    if isinstance(data, dict):
        for v in data.values():
            r = _find_list(v, depth + 1)
            if r is not None: return r
    return None

def parse_xml(file_path: str) -> tuple[pd.DataFrame, FileMetadata]:
    """Parse XML; finds repeating element and flattens attributes/nesting."""
    warnings: list[str] = []
    size_bytes = os.path.getsize(file_path)
    original_filename = os.path.basename(file_path)
    try:
        with open(file_path, "rb") as f:
            data = xmltodict.parse(f, attr_prefix="@", cdata_key="#text")
        records = _find_list(data)
        if records is None:
            warnings.append("No repeating element found; wrapping root")
            records = [data] if isinstance(data, dict) else []
        if not records:
            df = pd.DataFrame()
        else:
            flat = [_flatten(r) if isinstance(r, dict) else {"value": str(r)} for r in records]
            df = pd.DataFrame(flat).fillna("")
        logger.info(f"Parsed XML: {len(df)} rows x {len(df.columns)} cols")
    except Exception as e:
        warnings.append(f"XML parse error: {e}")
        logger.error(f"Failed to parse {file_path}: {e}")
        df = pd.DataFrame()
    return df, FileMetadata(
        original_filename=original_filename, file_format="xml",
        size_bytes=size_bytes, row_count=len(df), col_count=len(df.columns),
        parse_warnings=warnings,
    )
