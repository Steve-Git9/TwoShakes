"""CSV parser with auto-detection of encoding and delimiter."""

import os
import io
import logging
from typing import Optional

import chardet
import pandas as pd

from src.models.schemas import FileMetadata

logger = logging.getLogger(__name__)

DELIMITERS = [",", ";", "\t", "|"]


def _detect_encoding(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw = f.read()
    result = chardet.detect(raw)
    encoding = result.get("encoding") or "utf-8"
    logger.debug(f"Detected encoding: {encoding} (confidence {result.get('confidence', 0):.0%})")
    return encoding


def _detect_delimiter(file_path: str, encoding: str) -> str:
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        sample = f.read(4096)

    counts = {d: sample.count(d) for d in DELIMITERS}
    best = max(counts, key=counts.get)
    logger.debug(f"Detected delimiter: {repr(best)} (counts: {counts})")
    return best


def parse_csv(file_path: str) -> tuple[pd.DataFrame, FileMetadata]:
    """
    Parse a CSV file into a DataFrame with accompanying FileMetadata.

    Auto-detects encoding and delimiter. Handles errors gracefully by
    returning an empty DataFrame with warnings in the metadata.
    """
    warnings: list[str] = []
    size_bytes = os.path.getsize(file_path)
    original_filename = os.path.basename(file_path)

    try:
        encoding = _detect_encoding(file_path)
    except Exception as e:
        warnings.append(f"Encoding detection failed: {e}; defaulting to utf-8")
        encoding = "utf-8"

    try:
        delimiter = _detect_delimiter(file_path, encoding)
    except Exception as e:
        warnings.append(f"Delimiter detection failed: {e}; defaulting to comma")
        delimiter = ","

    df = pd.DataFrame()
    try:
        df = pd.read_csv(
            file_path,
            sep=delimiter,
            encoding=encoding,
            encoding_errors="replace",
            on_bad_lines="warn",
            dtype=str,          # keep everything as string — types detected later
        )
        logger.info(f"Parsed CSV: {len(df)} rows × {len(df.columns)} columns")
    except Exception as e:
        warnings.append(f"CSV parse error: {e}")
        logger.error(f"Failed to parse CSV {file_path}: {e}")

    metadata = FileMetadata(
        original_filename=original_filename,
        file_format="csv",
        encoding=encoding,
        size_bytes=size_bytes,
        row_count=len(df),
        col_count=len(df.columns),
        parse_warnings=warnings,
    )

    return df, metadata
