# DataPrepAgent — Technical Architecture

## Overview

DataPrepAgent is a 5-agent AI pipeline for automated data cleaning, built on Microsoft Azure AI Foundry and the Microsoft Agent Framework. It follows a **human-in-the-loop** design: the AI proposes, the human approves, deterministic code executes.

---

## Pipeline Data Flow

```
┌─────────────┐    FileMetadata + DataFrame
│   Upload    │ ─────────────────────────────► Ingestion Agent
└─────────────┘
                                                      │
                                               (df, FileMetadata)
                                                      │
                                                      ▼
                                              Profiler Agent
                                         (stats + LLM enrichment)
                                                      │
                                               ProfileReport
                                                      │
                                                      ▼
                                              Strategy Agent
                                           (LLM → CleaningPlan)
                                                      │
                                               CleaningPlan
                                                      │
                                                      ▼
                                           👤 Human Review
                                      (approve / reject / edit params)
                                                      │
                                          approved CleaningPlan
                                                      │
                                                      ▼
                                              Cleaner Agent
                                       (deterministic pandas transforms)
                                                      │
                                        (cleaned_df, TransformationLog)
                                                      │
                                                      ▼
                                             Validator Agent
                                      (6 checks + LLM certificate)
                                                      │
                                            ValidationReport
                                                      │
                                                      ▼
                                          ┌─────────────────┐
                                          │ Download Clean  │
                                          │ CSV/Excel/Parquet│
                                          └─────────────────┘
```

---

## Why "LLM Reasons, Python Executes"

This is the core architectural principle. The LLM is good at:
- Interpreting column semantics (`cust_nm` → customer name)
- Detecting cross-column quality issues
- Prioritising cleaning actions
- Writing natural-language explanations

Python pandas is good at:
- Reliable, reproducible data transformations
- Handling edge cases without hallucination
- Audit trails (exact rows affected)
- Speed on large datasets

By keeping them separate, we get the intelligence of LLM reasoning with the reliability of deterministic code. No AI-generated data values ever touch the output dataset.

---

## Agent Framework Integration

Each agent is backed by `AgentClient` in `src/agents/__init__.py`:

```python
class AgentClient:
    def __init__(self, name: str, instructions: str, json_mode: bool = False)
    async def run(self, message: str) -> str
```

At startup, `AgentClient.__init__` attempts:
1. `from agent_framework.azure import AzureAIChatClient` (Microsoft Agent Framework)
2. Falls back to `openai.AzureOpenAI` pointed at the Azure AI Foundry endpoint

This means the code runs correctly whether or not the Agent Framework preview package is installed, with no changes to calling code.

Three agents make LLM calls:
- **Profiler Agent** — semantic enrichment (JSON mode: `ProfileReport` fields)
- **Strategy Agent** — cleaning plan generation (JSON mode: `CleaningPlan`)
- **Validator Agent** — quality certificate (JSON mode: score + certificate text)

---

## Parser Architecture

```
parse_file(path)           ← dispatcher in src/parsers/__init__.py
    ├── .csv / .tsv  → parse_csv()    chardet encoding, auto-delimiter
    ├── .xlsx / .xls → parse_excel()  unmerge cells, multi-row headers
    ├── .json        → parse_json()   recursive list[dict] detection, json_normalize
    └── .xml         → parse_xml()    xmltodict, attribute flattening
```

All parsers return `(pd.DataFrame, FileMetadata)`. DataFrames are always `dtype=str` at ingest time — type conversion is handled later by the transformation layer.

---

## Transformation Layer

Each function in `src/transformations/` follows this contract:

```python
def transform_fn(df: pd.DataFrame, column: str, **params) -> tuple[pd.DataFrame, int]:
    """Returns (modified_copy, rows_affected). Never mutates input."""
```

This makes every transformation:
- **Auditable** — rows_affected is logged per action
- **Safe** — input DataFrame is never mutated
- **Testable** — pure function with no side effects

The Cleaner Agent dispatches to these functions via `_dispatch()` in `cleaner_agent.py`, wrapping each call in try/except so a single transform failure never crashes the pipeline.

---

## Azure Services Integration

| Service | Usage | Config key |
|---|---|---|
| Azure AI Foundry | LLM inference (GPT-4o) | `AZURE_AI_PROJECT_ENDPOINT` + `AZURE_AI_MODEL_DEPLOYMENT_NAME` |
| Azure AI Document Intelligence | OCR for scanned docs (optional) | `AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT` |
| Azure App Service (B1) | Web hosting | Configured via `infra/deploy.sh` |

The `AgentClient` uses `api_version="2024-12-01-preview"` to target the Azure AI Foundry OpenAI-compatible endpoint.

---

## Session State Management

The Streamlit app uses `st.session_state` as the single source of truth for pipeline state:

```python
st.session_state.uploaded_file      # raw file bytes
st.session_state.raw_df             # pd.DataFrame from parser
st.session_state.file_metadata      # FileMetadata
st.session_state.profile_report     # ProfileReport
st.session_state.cleaning_plan      # CleaningPlan (mutable — user edits)
st.session_state.cleaned_df         # pd.DataFrame after cleaning
st.session_state.transformation_log # TransformationLog
st.session_state.validation_report  # ValidationReport
st.session_state.current_step       # int 1–4 (drives sidebar indicator)
```

All Pydantic models are serialisable and can be persisted to disk or passed between sessions.

---

## Pydantic Data Contracts

All inter-agent data is typed via Pydantic v2 models in `src/models/schemas.py`:

```
FileMetadata        ← output of all parsers
ColumnProfile       ← per-column statistics + issues
ProfileReport       ← list[ColumnProfile] + overall score
CleaningAction      ← one proposed fix (type, column, params, priority, approved)
CleaningPlan        ← list[CleaningAction]
TransformationRecord ← one executed transform (success, rows_affected, error)
TransformationLog   ← list[TransformationRecord]
ValidationCheck     ← one check (name, passed, message)
ValidationReport    ← list[ValidationCheck] + score + certificate
```

LLM responses are parsed with `model.model_validate_json()` — if the LLM returns malformed JSON, the agent falls back to a safe default rather than crashing.

---

## Error Handling Strategy

| Layer | Strategy |
|---|---|
| Parser | Raises `ValueError` on unreadable files; caught by Upload page |
| Profiler | Statistical stage never fails; LLM stage falls back to empty semantic fields |
| Strategy | LLM JSON parse failure → empty plan with warning |
| Cleaner | Per-action try/except; failures logged, pipeline continues |
| Validator | Python checks always run; LLM certificate failure → placeholder text |
| Frontend | `st.stop()` on missing session state; `st.error()` on exceptions |

This ensures the app degrades gracefully: even if the LLM is unavailable, statistical profiling and deterministic cleaning still work.
