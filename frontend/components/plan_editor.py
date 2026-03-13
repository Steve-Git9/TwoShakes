"""Cleaning plan editor component (approve/reject individual actions)."""
import streamlit as st
from src.models.schemas import CleaningPlan


FILL_STRATEGIES = ["mean", "median", "mode", "ffill", "bfill", "custom", "drop"]
OUTLIER_ACTIONS = ["cap", "remove", "flag"]


def render_plan_editor(plan: CleaningPlan) -> CleaningPlan:
    """Render editable action cards; return updated plan."""
    for i, action in enumerate(plan.actions):
        priority = getattr(action, "priority", "medium")
        act_type = (
            action.action_type.value
            if hasattr(action.action_type, "value")
            else str(action.action_type)
        )
        col_name = action.column_name or "all rows"

        p_badge = f'<span class="priority-{priority}">{priority.upper()}</span>'
        header_col, toggle_col = st.columns([6, 1])
        with header_col:
            st.markdown(
                f"{p_badge} &nbsp; **{act_type}** &nbsp;—&nbsp; `{col_name}`",
                unsafe_allow_html=True,
            )
        with toggle_col:
            approved = st.toggle("", value=action.approved, key=f"action_{i}")
            plan.actions[i].approved = approved

        st.markdown(
            f'<span style="color:#666;font-size:0.88rem;">{action.description}</span>',
            unsafe_allow_html=True,
        )

        if act_type == "fill_missing" and approved:
            cur = action.parameters.get("strategy", "mean") if action.parameters else "mean"
            idx = FILL_STRATEGIES.index(cur) if cur in FILL_STRATEGIES else 0
            strat = st.selectbox("Fill strategy", FILL_STRATEGIES, index=idx, key=f"fs_{i}")
            if action.parameters is None:
                plan.actions[i].parameters = {}
            plan.actions[i].parameters["strategy"] = strat

        if act_type == "handle_outliers" and approved:
            cur = action.parameters.get("action", "cap") if action.parameters else "cap"
            idx = OUTLIER_ACTIONS.index(cur) if cur in OUTLIER_ACTIONS else 0
            out_a = st.selectbox("Outlier action", OUTLIER_ACTIONS, index=idx, key=f"oa_{i}")
            if action.parameters is None:
                plan.actions[i].parameters = {}
            plan.actions[i].parameters["action"] = out_a

        st.markdown("<hr style='margin:6px 0;border-color:#e0e0e0;'>", unsafe_allow_html=True)

    return plan
