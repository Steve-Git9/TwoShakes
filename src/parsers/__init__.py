"""Parser dispatcher."""
import os, logging
import pandas as pd
from src.models.schemas import FileMetadata
logger = logging.getLogger(__name__)
_EXT_MAP = {
    ".csv": "csv", ".tsv": "csv",
    ".xlsx": "excel", ".xls": "excel",
    ".json": "json", ".xml": "xml",
    ".pdf": "pdf",
}
def parse_file(file_path: str) -> tuple[pd.DataFrame, FileMetadata]:
    ext = os.path.splitext(file_path)[1].lower()
    fmt = _EXT_MAP.get(ext)
    if fmt is None:
        raise ValueError(f"Unsupported extension: {ext!r}")
    logger.info(f"Dispatching {os.path.basename(file_path)} -> {fmt} parser")
    if fmt == "csv":
        from src.parsers.csv_parser import parse_csv; return parse_csv(file_path)
    elif fmt == "excel":
        from src.parsers.excel_parser import parse_excel; return parse_excel(file_path)
    elif fmt == "json":
        from src.parsers.json_parser import parse_json; return parse_json(file_path)
    elif fmt == "xml":
        from src.parsers.xml_parser import parse_xml; return parse_xml(file_path)
    elif fmt == "pdf":
        from src.parsers.pdf_parser import parse_pdf; return parse_pdf(file_path)
