"""
Enterprise Audit Log
====================
Append-only JSONL audit trail for every DataPrepAgent pipeline run.

Design principles (Best Enterprise Solution):
  • No raw data is ever stored — only SHA-256 file hash and statistics
  • Append-only: records are never modified or deleted
  • Each event is a self-contained JSON line (easy to ship to SIEM / Splunk)
  • Captures who approved / rejected which actions (accountability)
  • Tracks LLM call count for cost governance

File location: audit.jsonl (project root by default; override via AUDIT_LOG_PATH env var)
"""

import hashlib
import logging
import os
from datetime import datetime
from pathlib import Path

from src.models.schemas import AuditEvent, CleaningPlan, FeatureEngineeringLog, ValidationReport

logger = logging.getLogger(__name__)

_DEFAULT_LOG_PATH = os.getenv("AUDIT_LOG_PATH", "audit.jsonl")


# ── Write ─────────────────────────────────────────────────────────────────────

def log_event(event: AuditEvent, log_path: str = _DEFAULT_LOG_PATH) -> None:
    """Append an AuditEvent as a single JSON line."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(event.model_dump_json() + "\n")
    logger.info(f"[Audit] Event {event.event_id[:8]}… written to {log_path}")


# ── Read ──────────────────────────────────────────────────────────────────────

def load_audit_log(log_path: str = _DEFAULT_LOG_PATH) -> list[AuditEvent]:
    """Load all audit events from the log file, newest first."""
    if not os.path.exists(log_path):
        return []
    events: list[AuditEvent] = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(AuditEvent.model_validate_json(line))
                except Exception as e:
                    logger.warning(f"[Audit] Skipping malformed log line: {e}")
    return list(reversed(events))


# ── Helpers ───────────────────────────────────────────────────────────────────

def hash_file(file_path: str) -> str:
    """Return the SHA-256 hex digest of a file (for audit identity, not content logging)."""
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


def log_feature_engineering(
    fe_log: FeatureEngineeringLog,
    col_count_before: int,
    col_count_after: int,
    row_count: int,
    log_path: str = _DEFAULT_LOG_PATH,
) -> None:
    """
    Append a feature engineering summary as a JSON line to the audit log.
    Logs approved/rejected action types and column-level stats.
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "event_type": "feature_engineering",
        "timestamp": datetime.utcnow().isoformat(),
        "row_count": row_count,
        "col_count_before": col_count_before,
        "col_count_after": col_count_after,
        "columns_added": fe_log.columns_added_total,
        "columns_removed": fe_log.columns_removed_total,
        "total_actions_executed": fe_log.total_actions_executed,
        "total_actions_succeeded": fe_log.total_actions_succeeded,
        "actions": [
            {
                "action_id": r.action_id,
                "action_type": r.action_type.value,
                "columns_affected": r.columns_affected,
                "columns_added": r.columns_added,
                "columns_removed": r.columns_removed,
                "success": r.success,
                "error": r.error_message,
            }
            for r in fe_log.records
        ],
    }
    with open(log_path, "a", encoding="utf-8") as f:
        import json
        f.write(json.dumps(entry) + "\n")
    logger.info(
        f"[Audit] FE event written: "
        f"{fe_log.total_actions_succeeded}/{fe_log.total_actions_executed} ok, "
        f"+{fe_log.columns_added_total}/-{fe_log.columns_removed_total} cols"
    )


def build_audit_event(
    file_path: str,
    file_format: str,
    cleaning_plan: CleaningPlan,
    validation_report: ValidationReport,
    backend_used: str,
    llm_calls_made: int,
    orchestration_attempts: int = 1,
) -> AuditEvent:
    """
    Construct an AuditEvent from pipeline outputs.
    Call this at the end of execute_approved_plan() before returning.
    """
    approved  = [a.action_type.value for a in cleaning_plan.actions if a.approved]
    rejected  = [a.action_type.value for a in cleaning_plan.actions if not a.approved]

    return AuditEvent(
        file_name=os.path.basename(file_path),
        file_hash=hash_file(file_path),
        file_format=file_format,
        row_count_before=validation_report.row_count_before,
        row_count_after=validation_report.row_count_after,
        quality_score_before=validation_report.before_quality_score,
        quality_score_after=validation_report.after_quality_score,
        improvement_pct=validation_report.improvement_percentage,
        total_actions=len(cleaning_plan.actions),
        approved_actions=len(approved),
        rejected_actions=len(rejected),
        rows_modified=validation_report.transformation_log.total_rows_modified,
        llm_calls_made=llm_calls_made,
        backend_used=backend_used,
        orchestration_attempts=orchestration_attempts,
    )
