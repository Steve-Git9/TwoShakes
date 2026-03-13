"""Profile Report page."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.ui_helpers import (
    load_css, render_sidebar, render_section_title, render_quality_gauge,
    render_quality_bar, render_empty_state, render_footer,
)

load_css()
render_sidebar()

profile = st.session_state.get("profile_report")

if profile is None:
    st.markdown(render_empty_state(
        "🔍",
        "No profile yet",
        "Upload and analyze a file first.",
    ), unsafe_allow_html=True)
    st.stop()

st.markdown(render_section_title("Data Profile Report", "AI-generated quality analysis of your dataset."), unsafe_allow_html=True)

# ── Top metrics ────────────────────────────────────────────────────────────────
meta = st.session_state.file_metadata
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{meta.row_count:,}")
c2.metric("Columns", meta.col_count)
c3.metric("Quality Score", f"{profile.overall_quality_score:.0f}/100")
missing_pct = sum(cp.missing.count for cp in profile.columns) / max(meta.row_count * meta.col_count, 1) * 100
c4.metric("Missing Values", f"{missing_pct:.1f}%")

st.divider()

# ── Quality gauge + Key issues ─────────────────────────────────────────────────
left_col, right_col = st.columns([1, 2])

with left_col:
    st.markdown(render_quality_gauge(profile.overall_quality_score), unsafe_allow_html=True)

with right_col:
    if profile.key_issues:
        st.markdown(render_section_title("Key Issues"), unsafe_allow_html=True)
        for issue in profile.key_issues:
            st.markdown(f'<span class="dp-issue-badge">⚠ {issue}</span>', unsafe_allow_html=True)
        st.write("")

st.divider()

# ── AI Summary ────────────────────────────────────────────────────────────────
with st.expander("AI Summary", expanded=True):
    st.write(profile.summary)

st.divider()

# ── Per-column profiles ───────────────────────────────────────────────────────
st.markdown(render_section_title("Column Profiles"), unsafe_allow_html=True)

TYPE_COLORS = {
    "numeric":     "#ac0000",
    "datetime":    "#8764b8",
    "boolean":     "#038387",
    "categorical": "#c19c00",
    "text":        "#2E7D32",
    "unknown":     "#737373",
}

for cp in profile.columns:
    col_type = cp.detected_type.value if hasattr(cp.detected_type, "value") else str(cp.detected_type)
    color = TYPE_COLORS.get(col_type, "#737373")
    badge = f'<span class="dp-col-type-badge" style="background:{color};">{col_type}</span>'
    with st.expander(cp.column_name, expanded=False):
        st.markdown(badge, unsafe_allow_html=True)
        left, right = st.columns([1, 1])
        with left:
            st.markdown(f"**Missing:** {cp.missing.count} ({cp.missing.percentage:.1f}%)")
            if cp.unique_count is not None:
                st.markdown(f"**Unique:** {cp.unique_count:,}")
            if cp.stats and cp.stats.get("mean") is not None:
                st.markdown(
                    f"**Mean:** {cp.stats['mean']:.3g} &nbsp;|&nbsp; "
                    f"**Std:** {cp.stats['std']:.3g}",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"**Min:** {cp.stats['min']} &nbsp;|&nbsp; **Max:** {cp.stats['max']}",
                    unsafe_allow_html=True,
                )
            q = cp.quality_score if cp.quality_score is not None else 100
            st.markdown(render_quality_bar(q), unsafe_allow_html=True)
        with right:
            if cp.top_values:
                st.markdown("**Top values:**")
                for v, c in list(cp.top_values.items())[:5]:
                    st.markdown(f"- `{v}` ({c}×)")
            if cp.sample_values:
                st.markdown(f"**Samples:** {', '.join(str(x) for x in cp.sample_values[:5])}")
            if cp.issues:
                for iss in cp.issues:
                    st.markdown(f'<span class="dp-issue-badge">{iss}</span>', unsafe_allow_html=True)

st.divider()

if profile.duplicates.exact_duplicate_count:
    st.warning(f"⚠ Duplicate rows detected: **{profile.duplicates.exact_duplicate_count}**")

if st.session_state.get("cleaning_plan"):
    st.markdown('<div class="dp-next-btn-container">', unsafe_allow_html=True)
    if st.button("Review Cleaning Plan →", type="primary", use_container_width=True):
        st.switch_page("pages/3_Cleaning_Plan.py")
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Upload a file first to generate a cleaning plan.")

st.markdown(render_footer(), unsafe_allow_html=True)
