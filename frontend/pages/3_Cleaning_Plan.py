"""Cleaning Plan review & execution page."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import asyncio
import concurrent.futures
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.ui_helpers import load_css, render_sidebar, render_section_title, render_empty_state, render_footer

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


plan = st.session_state.get("cleaning_plan")
df   = st.session_state.get("raw_df")

if plan is None or df is None:
    st.markdown(render_empty_state(
        "🧹",
        "No cleaning plan yet",
        "Upload and analyze a file first.",
    ), unsafe_allow_html=True)
    st.stop()

st.markdown(render_section_title("Cleaning Plan", "Review and approve AI-recommended cleaning actions."), unsafe_allow_html=True)

actions = plan.actions
total   = len(actions)
approved_count = sum(1 for a in actions if a.approved)
high_count = sum(
    1 for a in actions
    if (a.priority.value if hasattr(a.priority, "value") else str(a.priority)).lower() == "high"
)

# ── Summary bar ───────────────────────────────────────────────────────────────
sb1, sb2, sb3, sb4 = st.columns(4)
sb1.metric("Total Actions", total)
sb2.metric("Approved", approved_count)
sb3.metric("High Priority", high_count)
sb4.metric("Estimated Rows", f"{plan.estimated_rows_affected:,}" if hasattr(plan, 'estimated_rows_affected') and plan.estimated_rows_affected else "—")
st.divider()

# Select all / deselect all
sel_col1, sel_col2, _ = st.columns([1, 1, 4])
with sel_col1:
    if st.button("Select All"):
        for a in plan.actions:
            a.approved = True
        st.rerun()
with sel_col2:
    if st.button("Deselect All"):
        for a in plan.actions:
            a.approved = False
        st.rerun()

st.write("")

# ── Action cards ──────────────────────────────────────────────────────────────
FILL_STRATEGIES = ["mean", "median", "mode", "ffill", "bfill", "custom", "drop"]
OUTLIER_ACTIONS = ["cap", "remove", "flag"]

for i, action in enumerate(actions):
    priority = (action.priority.value if hasattr(action.priority, "value") else str(action.priority)).lower()
    act_type = action.action_type.value if hasattr(action.action_type, "value") else str(action.action_type)
    col_name = action.column_name or "all rows"
    badge_map = {"high": "dp-badge-high", "medium": "dp-badge-medium", "low": "dp-badge-low"}
    badge_cls = badge_map.get(priority, "dp-badge-info")

    with st.container():
        hdr_col, tog_col = st.columns([7, 1])
        with hdr_col:
            reason_safe = (action.reason or "").replace('"', '&quot;').replace("'", "&#39;")
            st.markdown(
                f'<div class="dp-action-header priority-{priority}">'
                f'<span class="{badge_cls}">{priority.upper()}</span>'
                f'<span class="dp-action-type">{act_type}</span>'
                f'<span class="dp-action-col">{col_name}</span>'
                f'<span class="dp-why-tooltip">Why?'
                f'  <span class="dp-tooltip-text">{reason_safe}</span>'
                f'</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="dp-action-desc-pad">{action.description}</div>',
                unsafe_allow_html=True,
            )
        with tog_col:
            approved = st.toggle("", value=action.approved, key=f"toggle_{i}", label_visibility="collapsed")
            plan.actions[i].approved = approved

        # Before preview
        if action.column_name and action.column_name in df.columns:
            samples = df[action.column_name].dropna().head(3).tolist()
            if samples:
                st.markdown(
                    f'<div class="dp-before-preview">Before: {" | ".join(str(s) for s in samples)}</div>',
                    unsafe_allow_html=True,
                )

        # Editable params (only if approved)
        if approved:
            if act_type == "fill_missing":
                strat = st.selectbox(
                    "Fill strategy",
                    FILL_STRATEGIES,
                    index=FILL_STRATEGIES.index(action.parameters.get("strategy", "mean"))
                          if action.parameters and action.parameters.get("strategy") in FILL_STRATEGIES else 0,
                    key=f"strat_{i}",
                )
                if action.parameters is None:
                    plan.actions[i].parameters = {}
                plan.actions[i].parameters["strategy"] = strat

            elif act_type == "handle_outlier":
                out_act = st.selectbox(
                    "Outlier action",
                    OUTLIER_ACTIONS,
                    index=OUTLIER_ACTIONS.index(action.parameters.get("action", "cap"))
                          if action.parameters and action.parameters.get("action") in OUTLIER_ACTIONS else 0,
                    key=f"outlier_{i}",
                )
                if action.parameters is None:
                    plan.actions[i].parameters = {}
                plan.actions[i].parameters["action"] = out_act

        st.markdown('<hr style="margin:6px 0;border-color:var(--border);">', unsafe_allow_html=True)

st.session_state.cleaning_plan = plan

st.divider()

# ── Execute / next-step ────────────────────────────────────────────────────────
if st.session_state.get("validation_report") is not None:
    # Cleaning already done — hide execute button, show success + CTA
    vr      = st.session_state.validation_report
    profile = st.session_state.profile_report
    q_before = profile.overall_quality_score if profile else 0
    st.markdown(f"""
<div class="dp-cert-box">
  <div class="dp-cert-title">✅ Cleaning Complete</div>
  <p class="dp-section-subtitle" style="margin:0.3rem 0 0 0;">
    Quality improved from <strong>{q_before:.0f}/100</strong>
    to <strong>{vr.after_quality_score:.0f}/100</strong>
    &nbsp;·&nbsp; <strong>{len(st.session_state.cleaned_df.columns)}</strong> columns retained
  </p>
</div>""", unsafe_allow_html=True)
    st.markdown('<div class="dp-next-btn-container">', unsafe_allow_html=True)
    if st.button("View Results & Download →", type="primary", use_container_width=True):
        st.switch_page("pages/4_Results.py")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    # Not yet executed — show the execute button
    approved_now = sum(1 for a in plan.actions if a.approved)
    if approved_now == 0:
        st.warning("No actions selected. Approve at least one action to proceed.")
        execute_disabled = True
    else:
        execute_disabled = False

    if st.button(
        f"Apply {approved_now} Cleaning Action{'s' if approved_now != 1 else ''}",
        type="primary",
        use_container_width=True,
        disabled=execute_disabled,
    ):
        from src.agents.cleaner_agent import execute_cleaning_plan
        from src.agents.validator_agent import validate

        progress = st.progress(0, text="Executing cleaning plan…")
        cleaned_df, tlog = _run(execute_cleaning_plan(df, plan))
        st.session_state.cleaned_df = cleaned_df
        st.session_state.transformation_log = tlog
        progress.progress(60, text="Validating results…")

        profile_r = st.session_state.profile_report
        vreport = _run(validate(df, cleaned_df, profile_r, tlog))
        st.session_state.validation_report = vreport
        st.session_state.current_step = max(st.session_state.current_step, 4)
        progress.progress(100, text="Done!")
        st.rerun()

st.markdown(render_footer(), unsafe_allow_html=True)
