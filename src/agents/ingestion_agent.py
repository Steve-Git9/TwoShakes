"""Ingestion Agent — parses any supported file and returns a clean raw DataFrame."""
import logging
import pandas as pd
from src.models.schemas import FileMetadata

logger = logging.getLogger(__name__)

async def ingest(file_path: str) -> tuple[pd.DataFrame, FileMetadata]:
    """Parse file via dispatcher, strip empty rows/cols, reset index."""
    from src.parsers import parse_file
    df, metadata = parse_file(file_path)
    if df.empty:
        return df, metadata

    df.columns = [str(c).strip() for c in df.columns]

    before_cols = len(df.columns)
    df = df.dropna(axis=1, how="all")
    str_empty = [c for c in df.columns if df[c].astype(str).str.strip().eq("").all()]
    df = df.drop(columns=str_empty) if str_empty else df
    if (d := before_cols - len(df.columns)):
        logger.info(f"Dropped {d} fully-empty column(s)")

    before_rows = len(df)
    df = df.dropna(how="all").reset_index(drop=True)
    all_empty = df.apply(lambda r: r.astype(str).str.strip().eq("").all(), axis=1)
    df = df[~all_empty].reset_index(drop=True)
    if (d := before_rows - len(df)):
        logger.info(f"Dropped {d} fully-empty row(s)")

    metadata = metadata.model_copy(update={"row_count": len(df), "col_count": len(df.columns)})
    logger.info(f"Ingestion complete: {len(df)} rows x {len(df.columns)} cols")
    return df, metadata

async def ingest_file(file_path: str) -> tuple[pd.DataFrame, FileMetadata]:
    """Backward-compatible alias."""
    return await ingest(file_path)
