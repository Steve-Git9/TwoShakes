"""Feature Engineering page — AI-recommended ML transformations."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import asyncio
import concurrent.futures

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.ui_helpers import load_css, render_sidebar, render_section_title, render_empty_state, render_data_preview, render_footer

load_css()
render_sidebar()


def _run(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                return pool.submit(asyncio.run, coro).result()
    except RuntimeError:
        pass
    return asyncio.run(coro)


# ── Guard ─────────────────────────────────────────────────────────────────────
cleaned = st.session_state.get("cleaned_df")
if cleaned is None or (hasattr(cleaned, "empty") and cleaned.empty):
    st.markdown(render_empty_state(
        "⚙",
        "No cleaned data found",
        "Complete the Cleaning Plan step first.",
    ), unsafe_allow_html=True)
    st.stop()

st.markdown(render_section_title(
    "Feature Engineering",
    "AI-recommended transformations to make your dataset ML-ready.",
), unsafe_allow_html=True)

target_col = st.session_state.get("fe_target_column")
if target_col:
    st.info(f"Target column: **{target_col}**")

# ── Step 1: Generate recommendations ──────────────────────────────────────────
if st.session_state.get("fe_plan") is None:
    if st.button("Analyze for ML Readiness", type="primary", use_container_width=True):
        with st.spinner("Feature Engineering Agent analyzing dataset…"):
            from src.agents.feature_engineering_agent import recommend_feature_engineering
            plan = _run(recommend_feature_engineering(cleaned, target_col))
            st.session_state.fe_plan = plan
        st.rerun()
    st.stop()

plan = st.session_state.fe_plan

# ── Summary bar ───────────────────────────────────────────────────────────────
total = len(plan.actions)
approved_count = sum(1 for a in plan.actions if a.approved)

c1, c2, c3 = st.columns(3)
c1.metric("Total Actions", total)
c2.metric("Approved", approved_count)
c3.metric("ML Task", plan.ml_task_hint or "general")
st.divider()

# ── Group actions by type ─────────────────────────────────────────────────────
ENCODING_TYPES  = {"one_hot_encode", "label_encode", "ordinal_encode", "target_encode", "frequency_encode"}
SCALING_TYPES   = {"min_max_scale", "standard_scale", "robust_scale", "max_abs_scale"}
DISTRIB_TYPES   = {"log_transform", "power_transform", "quantile_transform"}
CREATION_TYPES  = {"interaction_features", "polynomial_features", "binning"}
SELECTION_TYPES = {"drop_low_variance", "drop_high_cardinality", "drop_highly_correlated"}

GROUPS = [
    ("🔤 Encoding",          ENCODING_TYPES),
    ("📏 Scaling",           SCALING_TYPES),
    ("📊 Distribution",      DISTRIB_TYPES),
    ("🔧 Feature Creation",  CREATION_TYPES),
    ("✂ Feature Selection",  SELECTION_TYPES),
]

st.markdown(render_section_title("Review Recommended Transformations",
    "Toggle each action on/off. All are pre-approved — uncheck ones you want to skip."), unsafe_allow_html=True)

action_idx   = {a.id: i for i, a in enumerate(plan.actions)}
rendered_ids = set()

for group_label, group_types in GROUPS:
    group_actions = [
        a for a in plan.actions
        if (a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)) in group_types
    ]
    if not group_actions:
        continue

    st.markdown(f'<div class="dp-group-header">{group_label}</div>', unsafe_allow_html=True)

    for action in group_actions:
        rendered_ids.add(action.id)
        i = action_idx[action.id]
        col_label    = action.column_name or (", ".join(action.columns) if action.columns else "multiple columns")
        priority     = action.priority.value if hasattr(action.priority, "value") else str(action.priority)
        act_type_str = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        badge_map    = {"high": "dp-badge-high", "medium": "dp-badge-medium", "low": "dp-badge-low"}
        badge_cls    = badge_map.get(priority.lower(), "dp-badge-info")

        with st.container():
            hdr_col, toggle_col = st.columns([7, 1])
            with hdr_col:
                reason_safe = (action.reason or "").replace('"', '&quot;').replace("'", "&#39;")
                st.markdown(
                    f'<div class="dp-action-header priority-{priority.lower()}">'
                    f'<span class="{badge_cls}">{priority.upper()}</span>'
                    f'<span class="dp-action-type">{act_type_str}</span>'
                    f'<span class="dp-action-col">{col_label}</span>'
                    f'<span class="dp-why-tooltip">Why?'
                    f'  <span class="dp-tooltip-text">{reason_safe}</span>'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if action.description:
                    st.markdown(
                        f'<div class="dp-action-desc-pad">{action.description}</div>',
                        unsafe_allow_html=True,
                    )
                if action.warning:
                    st.warning(f"⚠ {action.warning}")
            with toggle_col:
                action.approved = st.toggle(
                    "Apply",
                    value=action.approved,
                    key=f"fe_toggle_{i}",
                    label_visibility="collapsed",
                )
            st.write("")

# Render any ungrouped actions
ungrouped = [a for a in plan.actions if a.id not in rendered_ids]
if ungrouped:
    st.markdown('<div class="dp-group-header">🔹 Other</div>', unsafe_allow_html=True)
    for action in ungrouped:
        i = action_idx[action.id]
        col_label    = action.column_name or (", ".join(action.columns) if action.columns else "multiple columns")
        priority     = action.priority.value if hasattr(action.priority, "value") else str(action.priority)
        act_type_str = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
        badge_map    = {"high": "dp-badge-high", "medium": "dp-badge-medium", "low": "dp-badge-low"}
        badge_cls    = badge_map.get(priority.lower(), "dp-badge-info")

        with st.container():
            hdr_col, toggle_col = st.columns([7, 1])
            with hdr_col:
                reason_safe = (action.reason or "").replace('"', '&quot;').replace("'", "&#39;")
                st.markdown(
                    f'<div class="dp-action-header priority-{priority.lower()}">'
                    f'<span class="{badge_cls}">{priority.upper()}</span>'
                    f'<span class="dp-action-type">{act_type_str}</span>'
                    f'<span class="dp-action-col">{col_label}</span>'
                    f'<span class="dp-why-tooltip">Why?'
                    f'  <span class="dp-tooltip-text">{reason_safe}</span>'
                    f'</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if action.description:
                    st.markdown(
                        f'<div class="dp-action-desc-pad">{action.description}</div>',
                        unsafe_allow_html=True,
                    )
            with toggle_col:
                action.approved = st.toggle(
                    "Apply",
                    value=action.approved,
                    key=f"fe_toggle_{i}",
                    label_visibility="collapsed",
                )
            st.write("")

st.divider()

# ── Execute / success state ───────────────────────────────────────────────────
if st.session_state.get("ml_ready_df") is not None:
    # Already executed — show success card, hide buttons
    ml_df_done = st.session_state.ml_ready_df
    st.markdown(f"""
<div class="dp-cert-box">
  <div class="dp-cert-title">✅ Feature Engineering Complete</div>
  <p class="dp-section-subtitle" style="margin:0.3rem 0 0 0;">
    <strong>{len(cleaned.columns)}</strong> columns →
    <strong>{len(ml_df_done.columns)}</strong> columns
    &nbsp;·&nbsp; <strong>{len(ml_df_done):,}</strong> rows ready for ML
  </p>
</div>""", unsafe_allow_html=True)
else:
    approved_now = sum(1 for a in plan.actions if a.approved)
    if approved_now == 0:
        st.warning("No actions selected. Approve at least one to proceed.")
        execute_disabled = True
    else:
        execute_disabled = False

    col_exec, col_skip = st.columns([3, 1])
    with col_exec:
        if st.button(
            f"Apply {approved_now} Feature Engineering Action{'s' if approved_now != 1 else ''}",
            type="primary",
            use_container_width=True,
            disabled=execute_disabled,
        ):
            progress = st.progress(0, text="Executing transformations…")
            from src.agents.feature_transformer_agent import execute_feature_engineering
            ml_df, fe_log = _run(execute_feature_engineering(cleaned, plan))
            progress.progress(80, text="Writing audit log…")

            try:
                from src.governance.audit_log import log_feature_engineering
                log_feature_engineering(
                    fe_log,
                    col_count_before=len(cleaned.columns),
                    col_count_after=len(ml_df.columns),
                    row_count=len(ml_df),
                )
            except Exception:
                pass

            st.session_state.ml_ready_df = ml_df
            st.session_state.fe_log = fe_log
            progress.progress(100, text="Done!")
            st.rerun()

    with col_skip:
        if st.button("Skip →", use_container_width=True):
            st.switch_page("pages/4_Results.py")

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.get("ml_ready_df") is not None:
    ml_df  = st.session_state.ml_ready_df
    fe_log = st.session_state.get("fe_log")

    st.divider()

    c1, c2, c3 = st.columns(3)
    c1.metric("Columns Before", len(cleaned.columns))
    c2.metric("Columns After", len(ml_df.columns), delta=len(ml_df.columns) - len(cleaned.columns))
    c3.metric("Object Columns Remaining", int(ml_df.select_dtypes("object").shape[1]))

    if fe_log and fe_log.records:
        with st.expander("Transformation Log", expanded=True):
            log_df = pd.DataFrame([
                {
                    "Action": r.action_id,
                    "Type": r.action_type.value,
                    "Columns Affected": ", ".join(r.columns_affected[:3]) + ("..." if len(r.columns_affected) > 3 else ""),
                    "Added": ", ".join(r.columns_added[:3]) + ("..." if len(r.columns_added) > 3 else ""),
                    "Removed": ", ".join(r.columns_removed[:3]) + ("..." if len(r.columns_removed) > 3 else ""),
                    "Status": "OK" if r.success else f"FAIL: {r.error_message}",
                }
                for r in fe_log.records
            ])
            st.markdown(render_data_preview(log_df, max_rows=50), unsafe_allow_html=True)

    st.markdown(render_section_title("ML-Ready Data Preview"), unsafe_allow_html=True)
    st.markdown(render_data_preview(ml_df, max_rows=10), unsafe_allow_html=True)

    with st.expander("Column dtypes"):
        dtype_df = pd.DataFrame({"Column": ml_df.dtypes.index, "Dtype": ml_df.dtypes.values.astype(str)})
        st.markdown(render_data_preview(dtype_df, max_rows=50), unsafe_allow_html=True)

    st.divider()
    st.markdown(render_section_title("Download ML-Ready Dataset"), unsafe_allow_html=True)
    import io
    d1, d2, d3 = st.columns(3)
    with d1:
        st.download_button(
            "Download CSV", ml_df.to_csv(index=False).encode("utf-8"),
            "ml_ready.csv", "text/csv", use_container_width=True,
        )
    with d2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            ml_df.to_excel(w, index=False, sheet_name="ML_Ready")
        st.download_button(
            "Download Excel", buf.getvalue(), "ml_ready.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
    with d3:
        pbuf = io.BytesIO()
        _pq = ml_df.copy()
        for _c in _pq.select_dtypes(include="object").columns:
            _pq[_c] = _pq[_c].astype(str)
        _pq.to_parquet(pbuf, index=False)
        st.download_button(
            "Download Parquet", pbuf.getvalue(), "ml_ready.parquet",
            "application/octet-stream", use_container_width=True,
        )

st.markdown(render_footer(), unsafe_allow_html=True)
