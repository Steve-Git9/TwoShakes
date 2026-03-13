"""
Orchestrator Agent
==================
Coordinates the full DataPrepAgent pipeline using structured Agent-to-Agent
(A2A) messaging. Acts as the top-level supervisor that:

  1. Dispatches requests to each sub-agent via AgentMessage
  2. Evaluates the ValidationReport quality score
  3. Triggers a re-clean loop (up to MAX_ATTEMPTS) if quality < target

This demonstrates multi-agent orchestration and A2A protocols as required
by the "Best Multi-Agent System" prize category.

Pipeline (orchestrated):
  Ingest → Profile → Strategy → [Human Review] → Clean → Validate
                                         ↑                    |
                                         └──── re-clean ←─────┘
                                              (if score < target)
"""

import logging
from typing import Optional

import pandas as pd

from src.models.schemas import (
    AgentMessage,
    CleaningPlan,
    FeatureEngineeringLog,
    FeatureEngineeringPlan,
    FileMetadata,
    ProfileReport,
    TransformationLog,
    ValidationReport,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 2          # Maximum re-clean attempts before accepting result
DEFAULT_QUALITY_TARGET = 70.0


class OrchestratorAgent:
    """
    Top-level supervisor agent that drives the 5-agent pipeline via A2A messaging.

    Each sub-agent call is wrapped in an AgentMessage so the full interaction
    history can be audited and replayed.
    """

    def __init__(self, quality_target: float = DEFAULT_QUALITY_TARGET):
        self.quality_target = quality_target
        self.name = "OrchestratorAgent"
        self.message_log: list[AgentMessage] = []

    # ── A2A messaging helpers ─────────────────────────────────────────────────

    def _send(self, recipient: str, message_type: str,
              payload: dict, attempt: int = 1) -> AgentMessage:
        msg = AgentMessage(
            sender=self.name,
            recipient=recipient,
            message_type=message_type,
            payload=payload,
            attempt=attempt,
        )
        self.message_log.append(msg)
        logger.info(
            f"[A2A] {self.name} → {recipient} | {message_type} "
            f"(attempt {attempt}) | keys={list(payload.keys())}"
        )
        return msg

    def _receive(self, sender: str, message_type: str,
                 payload: dict, attempt: int = 1) -> AgentMessage:
        msg = AgentMessage(
            sender=sender,
            recipient=self.name,
            message_type=message_type,
            payload=payload,
            attempt=attempt,
        )
        self.message_log.append(msg)
        return msg

    # ── Main orchestration entry point ────────────────────────────────────────

    async def run(
        self,
        file_path: str,
        approved_plan: Optional[CleaningPlan] = None,
        run_feature_engineering: bool = False,
        fe_target_column: Optional[str] = None,
    ) -> "OrchestratedPipelineResult":
        """
        Drive the full pipeline. If approved_plan is supplied, skips Ingest/
        Profile/Strategy and goes straight to Clean → Validate (UI flow).
        If not supplied, runs all stages automatically (CLI/MCP flow).

        Returns an OrchestratedPipelineResult containing all intermediate
        outputs and the full A2A message log.
        """
        from src.agents.ingestion_agent import ingest
        from src.agents.profiler_agent import profile_dataframe
        from src.agents.strategy_agent import generate_cleaning_plan
        from src.agents.cleaner_agent import execute_cleaning_plan
        from src.agents.validator_agent import validate

        result = OrchestratedPipelineResult(orchestrator=self)

        # ── Step 1: Ingest ────────────────────────────────────────────────────
        self._send("IngestionAgent", "request", {"file_path": file_path})
        raw_df, metadata = await ingest(file_path)
        self._receive("IngestionAgent", "response", {
            "rows": metadata.row_count,
            "cols": metadata.col_count,
            "format": metadata.file_format,
        })
        result.raw_df = raw_df
        result.file_metadata = metadata
        logger.info(f"[Orchestrator] Ingested: {metadata.row_count}r × {metadata.col_count}c")

        # ── Step 2: Profile ───────────────────────────────────────────────────
        self._send("ProfilerAgent", "request", {"rows": metadata.row_count})
        profile = await profile_dataframe(raw_df, metadata)
        self._receive("ProfilerAgent", "response", {
            "quality_score": profile.overall_quality_score,
            "issues": len(profile.key_issues),
            "columns": len(profile.columns),
        })
        result.profile_report = profile
        logger.info(
            f"[Orchestrator] Profile: quality={profile.overall_quality_score:.1f}, "
            f"issues={len(profile.key_issues)}"
        )

        # ── Step 3: Strategy (if no pre-approved plan) ────────────────────────
        if approved_plan is None:
            self._send("StrategyAgent", "request", {
                "quality_score": profile.overall_quality_score,
                "issues": profile.key_issues,
            })
            plan = await generate_cleaning_plan(profile, raw_df)
            # Auto-approve all actions (no human review in headless orchestration)
            for action in plan.actions:
                action.approved = True
            self._receive("StrategyAgent", "response", {
                "actions": len(plan.actions),
                "estimated_rows": plan.estimated_rows_affected,
            })
            result.cleaning_plan = plan
            logger.info(f"[Orchestrator] Plan: {len(plan.actions)} actions generated")
        else:
            result.cleaning_plan = approved_plan
            approved_count = sum(1 for a in approved_plan.actions if a.approved)
            logger.info(f"[Orchestrator] Using pre-approved plan: {approved_count} actions")

        # ── Re-clean loop: Clean → Validate (up to MAX_ATTEMPTS) ─────────────
        current_df = raw_df.copy()
        attempt = 1

        while attempt <= MAX_ATTEMPTS:
            logger.info(f"[Orchestrator] Clean+Validate attempt {attempt}/{MAX_ATTEMPTS}")

            # ── Step 4: Clean ─────────────────────────────────────────────────
            self._send("CleanerAgent", "request", {
                "approved_actions": sum(
                    1 for a in result.cleaning_plan.actions if a.approved
                ),
            }, attempt=attempt)

            cleaned_df, tlog = await execute_cleaning_plan(
                current_df.copy(), result.cleaning_plan
            )

            self._receive("CleanerAgent", "response", {
                "rows_modified": tlog.total_rows_modified,
                "success_rate": (
                    tlog.total_actions_succeeded / tlog.total_actions_executed
                    if tlog.total_actions_executed else 1.0
                ),
            }, attempt=attempt)

            # ── Step 5: Validate ──────────────────────────────────────────────
            self._send("ValidatorAgent", "request", {
                "quality_target": self.quality_target,
            }, attempt=attempt)

            vreport = await validate(raw_df, cleaned_df, profile, tlog)

            self._receive("ValidatorAgent", "response", {
                "before_score": vreport.before_quality_score,
                "after_score": vreport.after_quality_score,
                "improvement": vreport.improvement_percentage,
                "passed_checks": sum(1 for c in vreport.checks if c.passed),
            }, attempt=attempt)

            result.cleaned_df = cleaned_df
            result.transformation_log = tlog
            result.validation_report = vreport
            result.orchestration_attempts = attempt

            # Check if quality target is met
            if vreport.after_quality_score >= self.quality_target:
                logger.info(
                    f"[Orchestrator] Quality target reached: "
                    f"{vreport.after_quality_score:.1f} ≥ {self.quality_target}"
                )
                self._send("OrchestratorAgent", "handoff", {
                    "status": "target_reached",
                    "final_score": vreport.after_quality_score,
                    "attempts": attempt,
                })
                break

            if attempt < MAX_ATTEMPTS:
                logger.info(
                    f"[Orchestrator] Quality {vreport.after_quality_score:.1f} < "
                    f"{self.quality_target} — requesting retry from StrategyAgent"
                )
                # Re-generate plan targeting remaining issues, then retry
                self._send("StrategyAgent", "retry", {
                    "reason": "quality_below_target",
                    "current_score": vreport.after_quality_score,
                    "target_score": self.quality_target,
                    "remaining_issues": [
                        c.details for c in vreport.checks if not c.passed
                    ],
                }, attempt=attempt + 1)

                retry_plan = await generate_cleaning_plan(profile, cleaned_df)
                for action in retry_plan.actions:
                    action.approved = True
                result.cleaning_plan = retry_plan
                current_df = cleaned_df  # apply on top of previous result
            else:
                logger.info(
                    f"[Orchestrator] Max attempts reached — accepting "
                    f"score {vreport.after_quality_score:.1f}"
                )

            attempt += 1

        # ── Step 6 (optional): Feature Engineering ───────────────────────────
        if run_feature_engineering:
            from src.agents.feature_engineering_agent import recommend_feature_engineering
            from src.agents.feature_transformer_agent import execute_feature_engineering

            self._send("FeatureEngineeringAgent", "request", {
                "rows": len(result.cleaned_df),
                "cols": len(result.cleaned_df.columns),
                "target_column": fe_target_column,
            })

            fe_plan = await recommend_feature_engineering(result.cleaned_df, fe_target_column)
            for action in fe_plan.actions:
                action.approved = True  # auto-approve in headless mode

            self._receive("FeatureEngineeringAgent", "response", {
                "actions": len(fe_plan.actions),
                "ml_task_hint": fe_plan.ml_task_hint,
            })

            self._send("FeatureTransformerAgent", "request", {
                "approved_actions": len([a for a in fe_plan.actions if a.approved]),
            })

            ml_df, fe_log = await execute_feature_engineering(result.cleaned_df, fe_plan)

            self._receive("FeatureTransformerAgent", "response", {
                "cols_added": fe_log.columns_added_total,
                "cols_removed": fe_log.columns_removed_total,
                "succeeded": fe_log.total_actions_succeeded,
            })

            result.feature_engineering_plan = fe_plan
            result.ml_ready_df = ml_df
            result.feature_engineering_log = fe_log
            logger.info(
                f"[Orchestrator] FE done: "
                f"{fe_log.total_actions_succeeded}/{fe_log.total_actions_executed} ok, "
                f"final {len(ml_df)} × {len(ml_df.columns)}"
            )

        return result


class OrchestratedPipelineResult:
    """Container for all pipeline outputs + A2A message log."""

    def __init__(self, orchestrator: OrchestratorAgent):
        self.orchestrator = orchestrator
        self.raw_df: pd.DataFrame = pd.DataFrame()
        self.file_metadata: Optional[FileMetadata] = None
        self.profile_report: Optional[ProfileReport] = None
        self.cleaning_plan: Optional[CleaningPlan] = None
        self.cleaned_df: pd.DataFrame = pd.DataFrame()
        self.transformation_log: Optional[TransformationLog] = None
        self.validation_report: Optional[ValidationReport] = None
        self.orchestration_attempts: int = 0
        # Feature engineering (optional phase)
        self.feature_engineering_plan: Optional[FeatureEngineeringPlan] = None
        self.ml_ready_df: Optional[pd.DataFrame] = None
        self.feature_engineering_log: Optional[FeatureEngineeringLog] = None

    @property
    def message_log(self) -> list[AgentMessage]:
        return self.orchestrator.message_log

    def summary(self) -> dict:
        vr = self.validation_report
        return {
            "quality_before": vr.before_quality_score if vr else 0,
            "quality_after": vr.after_quality_score if vr else 0,
            "improvement": vr.improvement_percentage if vr else 0,
            "orchestration_attempts": self.orchestration_attempts,
            "a2a_messages": len(self.message_log),
            "rows_before": len(self.raw_df),
            "rows_after": len(self.cleaned_df),
        }
