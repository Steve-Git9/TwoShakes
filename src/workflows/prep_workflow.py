"""
DataPrepAgent pipeline orchestration.

run_prep_workflow()    — runs Ingest + Profile + Strategy, returns state before human review.
execute_approved_plan() — runs Cleaner + Validator on the approved plan + writes audit log.
"""
import logging
from dataclasses import dataclass, field
import pandas as pd
from src.agents.ingestion_agent import ingest
from src.agents.profiler_agent import profile_dataframe
from src.agents.strategy_agent import generate_cleaning_plan
from src.agents.cleaner_agent import execute_cleaning_plan
from src.agents.validator_agent import validate
from src.models.schemas import CleaningPlan, FileMetadata, ProfileReport, ValidationReport, TransformationLog
from src.governance.audit_log import build_audit_event, log_event

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    file_path: str
    raw_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    file_metadata: FileMetadata | None = None
    profile_report: ProfileReport | None = None
    cleaning_plan: CleaningPlan | None = None
    cleaned_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    transformation_log: TransformationLog | None = None
    validation_report: ValidationReport | None = None


async def run_prep_workflow(file_path: str) -> PipelineState:
    """
    Run: Ingest -> Profile -> Strategy.
    Returns state with cleaning_plan populated but NOT yet executed.
    The Streamlit UI shows the plan for human review before calling execute_approved_plan().
    """
    state = PipelineState(file_path=file_path)

    logger.info("=== Step 1: Ingestion ===")
    state.raw_df, state.file_metadata = await ingest(file_path)

    logger.info("=== Step 2: Profiling ===")
    state.profile_report = await profile_dataframe(state.raw_df, state.file_metadata)

    logger.info("=== Step 3: Strategy ===")
    state.cleaning_plan = await generate_cleaning_plan(state.profile_report, state.raw_df)

    return state


async def execute_approved_plan(state: PipelineState) -> PipelineState:
    """
    Run: Clean -> Validate on the (possibly user-modified) cleaning_plan in state.
    Mutates and returns state.
    """
    if state.cleaning_plan is None:
        raise ValueError("No cleaning plan to execute. Call run_prep_workflow() first.")

    logger.info("=== Step 4: Cleaning ===")
    state.cleaned_df, state.transformation_log = await execute_cleaning_plan(
        state.raw_df.copy(), state.cleaning_plan
    )

    logger.info("=== Step 5: Validation ===")
    state.validation_report = await validate(
        state.raw_df, state.cleaned_df, state.profile_report, state.transformation_log
    )

    # ── Audit log ─────────────────────────────────────────────────────────────
    try:
        from src.agents import _AGENT_FRAMEWORK_AVAILABLE
        backend = "agent-framework" if _AGENT_FRAMEWORK_AVAILABLE else "openai-sdk"
        # 3 LLM calls: profiler enrichment + strategy + validator certificate
        audit_event = build_audit_event(
            file_path=state.file_path,
            file_format=state.file_metadata.file_format if state.file_metadata else "unknown",
            cleaning_plan=state.cleaning_plan,
            validation_report=state.validation_report,
            backend_used=backend,
            llm_calls_made=3,
        )
        log_event(audit_event)
    except Exception as e:
        logger.warning(f"Audit log write failed (non-fatal): {e}")

    return state


async def run_pipeline(file_path: str, auto_approve: bool = True) -> PipelineState:
    """
    Convenience: run full pipeline end-to-end (for tests / CLI).
    If auto_approve=True all actions are approved automatically.
    """
    state = await run_prep_workflow(file_path)
    if auto_approve:
        for action in state.cleaning_plan.actions:
            action.approved = True
    return await execute_approved_plan(state)
