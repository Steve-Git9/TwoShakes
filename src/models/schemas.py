from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Any
from datetime import datetime
import uuid


class FileMetadata(BaseModel):
    """Metadata about an uploaded file after initial parsing."""
    original_filename: str
    file_format: str                          # csv, xlsx, json, xml
    encoding: Optional[str] = None
    size_bytes: int
    sheet_names: Optional[list[str]] = None   # Excel only
    row_count: int
    col_count: int
    parse_warnings: list[str] = Field(default_factory=list)


class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    TEXT = "text"
    ID = "id"
    UNKNOWN = "unknown"


class MissingValueInfo(BaseModel):
    count: int
    percentage: float
    encoded_as: list[str]                     # e.g. ["N/A", "null", "", "-"]


class ColumnProfile(BaseModel):
    """Profile of a single column."""
    column_name: str
    detected_type: ColumnType
    inferred_semantic_type: Optional[str] = None   # e.g. "email", "currency", "country"
    missing: MissingValueInfo
    unique_count: int
    cardinality_ratio: float
    sample_values: list[str]
    stats: Optional[dict[str, Any]] = None         # numeric stats
    top_values: Optional[dict[str, int]] = None    # categorical value counts
    issues: list[str] = Field(default_factory=list)
    quality_score: float                           # 0-100


class DuplicateInfo(BaseModel):
    exact_duplicate_count: int
    exact_duplicate_row_indices: list[list[int]]
    fuzzy_duplicate_count: int
    fuzzy_duplicate_groups: list[dict[str, Any]]


class ProfileReport(BaseModel):
    """Complete data quality profile report."""
    file_metadata: FileMetadata
    columns: list[ColumnProfile]
    duplicates: DuplicateInfo
    overall_quality_score: float               # 0-100
    summary: str                               # LLM-generated natural language
    key_issues: list[str]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class ActionType(str, Enum):
    CONVERT_TYPE = "convert_type"
    FILL_MISSING = "fill_missing"
    STANDARDIZE_CATEGORICAL = "standardize_categorical"
    PARSE_DATES = "parse_dates"
    REMOVE_DUPLICATES = "remove_duplicates"
    HANDLE_OUTLIER = "handle_outlier"
    FIX_ENCODING = "fix_encoding"
    RENAME_COLUMN = "rename_column"
    DROP_COLUMN = "drop_column"
    TRIM_WHITESPACE = "trim_whitespace"
    NORMALIZE_CASE = "normalize_case"


class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class CleaningAction(BaseModel):
    """A single proposed cleaning action."""
    id: str
    column_name: Optional[str] = None          # None for row-level actions like dedup
    action_type: ActionType
    description: str
    parameters: dict[str, Any]
    reason: str
    priority: Priority
    approved: bool = True                      # user toggles this in the UI


class CleaningPlan(BaseModel):
    """Complete cleaning plan: a list of actions to review and execute."""
    actions: list[CleaningAction]
    estimated_rows_affected: int
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class TransformationRecord(BaseModel):
    """Log of a single executed transformation."""
    action_id: str
    column_name: Optional[str]
    action_type: ActionType
    rows_affected: int
    before_sample: list[Any]
    after_sample: list[Any]
    success: bool
    error_message: Optional[str] = None


class TransformationLog(BaseModel):
    records: list[TransformationRecord]
    total_actions_executed: int
    total_actions_succeeded: int
    total_rows_modified: int


class ValidationCheck(BaseModel):
    check_name: str
    passed: bool
    details: str
    severity: str                              # "info", "warning", "error"


class ValidationReport(BaseModel):
    """Final quality report after cleaning."""
    checks: list[ValidationCheck]
    before_quality_score: float
    after_quality_score: float
    improvement_percentage: float
    data_quality_certificate: str              # LLM-generated
    transformation_log: TransformationLog
    row_count_before: int
    row_count_after: int
    col_count_before: int
    col_count_after: int


# ── Feature Engineering models ───────────────────────────────────────────────

class FeatureEngineeringActionType(str, Enum):
    """All supported feature engineering transformation types."""
    ONE_HOT_ENCODE = "one_hot_encode"
    LABEL_ENCODE = "label_encode"
    ORDINAL_ENCODE = "ordinal_encode"
    TARGET_ENCODE = "target_encode"
    FREQUENCY_ENCODE = "frequency_encode"
    MIN_MAX_SCALE = "min_max_scale"
    STANDARD_SCALE = "standard_scale"
    ROBUST_SCALE = "robust_scale"
    MAX_ABS_SCALE = "max_abs_scale"
    LOG_TRANSFORM = "log_transform"
    POWER_TRANSFORM = "power_transform"
    QUANTILE_TRANSFORM = "quantile_transform"
    BINNING = "binning"
    INTERACTION_FEATURES = "interaction_features"
    POLYNOMIAL_FEATURES = "polynomial_features"
    DROP_LOW_VARIANCE = "drop_low_variance"
    DROP_HIGH_CARDINALITY = "drop_high_cardinality"
    DROP_HIGHLY_CORRELATED = "drop_highly_correlated"


class FeatureEngineeringAction(BaseModel):
    """A single proposed feature engineering action."""
    id: str
    column_name: Optional[str] = None
    columns: Optional[list[str]] = None
    action_type: FeatureEngineeringActionType
    description: str
    parameters: dict[str, Any]
    reason: str
    impact: str
    priority: Priority
    approved: bool = True
    warning: Optional[str] = None


class FeatureEngineeringPlan(BaseModel):
    """Complete feature engineering plan."""
    target_column: Optional[str] = None
    ml_task_hint: Optional[str] = None
    actions: list[FeatureEngineeringAction]
    generated_at: datetime = Field(default_factory=datetime.utcnow)


class FeatureEngineeringRecord(BaseModel):
    """Log of a single executed feature engineering transformation."""
    action_id: str
    action_type: FeatureEngineeringActionType
    columns_affected: list[str]
    columns_added: list[str]
    columns_removed: list[str]
    rows_before: int
    rows_after: int
    cols_before: int
    cols_after: int
    success: bool
    error_message: Optional[str] = None


class FeatureEngineeringLog(BaseModel):
    records: list[FeatureEngineeringRecord]
    total_actions_executed: int
    total_actions_succeeded: int
    columns_added_total: int
    columns_removed_total: int


# ── Agent-to-Agent (A2A) messaging ───────────────────────────────────────────

class AgentMessage(BaseModel):
    """
    Structured message passed between agents in the orchestrated pipeline.
    Enables explicit A2A communication and audit trail.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str                                     # e.g. "OrchestratorAgent"
    recipient: str                                  # e.g. "ProfilerAgent"
    message_type: str                               # "request" | "response" | "handoff" | "retry"
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    attempt: int = 1                                # retry count


# ── Enterprise audit event ───────────────────────────────────────────────────

class AuditEvent(BaseModel):
    """
    Immutable audit record written per pipeline run.
    Never stores raw data — only hashes and statistics.
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    file_name: str
    file_hash: str                                  # SHA-256 of original file
    file_format: str
    row_count_before: int
    row_count_after: int
    quality_score_before: float
    quality_score_after: float
    improvement_pct: float
    total_actions: int
    approved_actions: int
    rejected_actions: int
    rows_modified: int
    llm_calls_made: int
    backend_used: str                               # "agent-framework" | "openai-sdk"
    orchestration_attempts: int = 1
