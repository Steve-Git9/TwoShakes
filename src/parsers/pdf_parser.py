"""
PDF Parser — Azure Document Intelligence
=========================================
Extracts tabular data from PDF files (including scanned/image PDFs) using
the Azure AI Document Intelligence prebuilt-layout model.

This is the only parser that requires an Azure service credential.
If the service is unavailable or credentials are missing, a clear error
is raised so the user knows exactly what to configure.

Required environment variables:
    AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT
    AZURE_DOCUMENT_INTELLIGENCE_KEY
"""

import os
import logging
from typing import Optional

import pandas as pd

from src.models.schemas import FileMetadata

logger = logging.getLogger(__name__)


def parse_pdf(
    file_path: str,
    table_index: int = 0,
) -> tuple[pd.DataFrame, FileMetadata]:
    """
    Extract a table from a PDF using Azure Document Intelligence.

    Args:
        file_path:   Path to the PDF file.
        table_index: Which table to extract when the PDF has multiple (default: 0).

    Returns:
        (DataFrame, FileMetadata) — same contract as all other parsers.

    Raises:
        EnvironmentError: If AZURE_DOCUMENT_INTELLIGENCE_* env vars are missing.
        ValueError:       If the PDF contains no tables.
        ImportError:      If azure-ai-documentintelligence is not installed.
    """
    try:
        from azure.ai.documentintelligence import DocumentIntelligenceClient  # type: ignore
        from azure.core.credentials import AzureKeyCredential               # type: ignore
    except ImportError as exc:
        raise ImportError(
            "azure-ai-documentintelligence is required for PDF parsing. "
            "Install it with: pip install azure-ai-documentintelligence"
        ) from exc

    endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "")
    api_key  = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "")

    if not endpoint or not api_key:
        raise EnvironmentError(
            "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY "
            "must be set in .env to parse PDF files."
        )

    file_size = os.path.getsize(file_path)
    logger.info(f"Parsing PDF via Azure Document Intelligence: {file_path}")

    client = DocumentIntelligenceClient(
        endpoint=endpoint,
        credential=AzureKeyCredential(api_key),
    )

    with open(file_path, "rb") as f:
        poller = client.begin_analyze_document(
            "prebuilt-layout",
            analyze_request=f,
            content_type="application/octet-stream",
        )

    result = poller.result()

    if not result.tables:
        raise ValueError(
            f"No tables found in {os.path.basename(file_path)}. "
            "Azure Document Intelligence could not extract tabular data from this PDF."
        )

    # Select the requested table
    if table_index >= len(result.tables):
        logger.warning(
            f"table_index={table_index} out of range "
            f"({len(result.tables)} tables found) — using table 0"
        )
        table_index = 0

    table = result.tables[table_index]
    n_rows = table.row_count
    n_cols = table.column_count

    logger.info(
        f"Extracted table {table_index}: {n_rows} rows × {n_cols} cols "
        f"({len(result.tables)} tables total in PDF)"
    )

    # Build a 2-D grid from the cell list
    grid: list[list[str]] = [[""] * n_cols for _ in range(n_rows)]
    for cell in table.cells:
        grid[cell.row_index][cell.column_index] = (cell.content or "").strip()

    # Row 0 becomes the header
    headers = grid[0] if grid else []
    # Deduplicate blank / duplicate header names
    seen: dict[str, int] = {}
    clean_headers: list[str] = []
    for h in headers:
        if not h:
            h = f"col_{len(clean_headers)}"
        if h in seen:
            seen[h] += 1
            h = f"{h}_{seen[h]}"
        else:
            seen[h] = 0
        clean_headers.append(h)

    data_rows = grid[1:] if len(grid) > 1 else []
    df = pd.DataFrame(data_rows, columns=clean_headers, dtype=str)

    # Drop fully empty rows / columns (common in PDFs with merged cells)
    df = df.replace("", pd.NA).dropna(how="all").dropna(axis=1, how="all")
    df = df.fillna("").reset_index(drop=True)

    warnings = []
    if len(result.tables) > 1:
        warnings.append(
            f"PDF contains {len(result.tables)} tables; extracted table {table_index}. "
            "Pass table_index= to select a different table."
        )

    metadata = FileMetadata(
        original_filename=os.path.basename(file_path),
        file_format="pdf",
        size_bytes=file_size,
        row_count=len(df),
        col_count=len(df.columns),
        parse_warnings=warnings,
    )

    logger.info(
        f"PDF parse complete: {metadata.row_count} rows × {metadata.col_count} cols"
    )
    return df, metadata
