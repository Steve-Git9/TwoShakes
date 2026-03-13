"""Profile summary cards and column profile renderer."""
import streamlit as st
from src.models.schemas import ProfileReport, ColumnProfile


TYPE_COLORS = {
    "numeric": "#0078d4",
    "datetime": "#8764b8",
    "boolean": "#038387",
    "categorical": "#c19c00",
    "text": "#107c10",
    "unknown": "#737373",
}


def render_profile_summary(profile: ProfileReport):
    """Top-level metric cards + key issues + LLM summary."""
    st.metric("Overall Quality", f"{profile.overall_quality_score:.0f}/100")
    st.write(profile.summary)
    if profile.key_issues:
        for issue in profile.key_issues:
            st.markdown(
                f'<span class="issue-badge">⚠ {issue}</span>',
                unsafe_allow_html=True,
            )


def render_column_profile(cp: ColumnProfile):
    """Render a single column profile inside an expander."""
    col_type = (
        cp.inferred_type.value
        if hasattr(cp.inferred_type, "value")
        else str(cp.inferred_type)
    )
    color = TYPE_COLORS.get(col_type, "#737373")
    badge = (
        f'<span style="background:{color};color:#fff;padding:2px 8px;' +
        f'border-radius:10px;font-size:0.78rem;font-weight:600;">{col_type}</span>'
    )
    with st.expander(f"{cp.column_name}  {badge}", expanded=False):
        left, right = st.columns(2)
        with left:
            st.markdown(f"**Type:** {col_type}")
            st.markdown(f"**Missing:** {cp.missing_count} ({cp.missing_pct:.1f}%)")
            if cp.unique_count is not None:
                st.markdown(f"**Unique:** {cp.unique_count}")
            if cp.mean is not None:
                st.markdown(f"**Mean:** {cp.mean:.3g}  |  **Std:** {cp.std:.3g}")
        with right:
            if cp.top_values:
                st.markdown("**Top values:**")
                for v, c in list(cp.top_values.items())[:5]:
                    st.markdown(f"- `{v}` ({c}×)")
            if cp.issues:
                for iss in cp.issues:
                    st.markdown(
                        f'<span class="issue-badge">{iss}</span>',
                        unsafe_allow_html=True,
                    )
