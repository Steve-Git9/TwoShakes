"""Results & Download page."""
import sys, os, io
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.ui_helpers import (
    load_css, render_sidebar, render_section_title, render_empty_state,
    render_comparison_metric, render_data_preview, render_footer,
)

load_css()
render_sidebar()

vreport = st.session_state.get("validation_report")
cleaned = st.session_state.get("cleaned_df")
original = st.session_state.get("raw_df")
tlog    = st.session_state.get("transformation_log")
profile = st.session_state.get("profile_report")

if vreport is None or cleaned is None:
    st.markdown(render_empty_state(
        "✅",
        "No results yet",
        "Complete the cleaning plan step first.",
    ), unsafe_allow_html=True)
    st.stop()

st.markdown(render_section_title("Results", "Your data has been cleaned and validated."), unsafe_allow_html=True)

# ── Before/after metrics ───────────────────────────────────────────────────────
score_before = profile.overall_quality_score if profile else 0
score_after  = vreport.after_quality_score
delta_score  = score_after - score_before
rows_before  = len(original) if original is not None else 0
rows_after   = len(cleaned)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Quality Before", f"{score_before:.0f}/100")
c2.metric("Quality After",  f"{score_after:.0f}/100", delta=f"{delta_score:+.0f}")
c3.metric("Rows Before",    f"{rows_before:,}")
c4.metric("Rows After",     f"{rows_after:,}", delta=f"{rows_after - rows_before:+,}")

st.divider()

# ── Validation checks ──────────────────────────────────────────────────────────
st.markdown(render_section_title("Validation Checks"), unsafe_allow_html=True)
for check in vreport.checks:
    icon = "✅" if check.passed else "❌"
    st.markdown(f"{icon} **{check.check_name}** — {check.details}")

st.divider()

# ── Data Quality Certificate ───────────────────────────────────────────────────
if vreport.data_quality_certificate:
    st.markdown(f"""
<div class="dp-cert-box">
  <div class="dp-cert-title">🏆 Data Quality Certificate</div>
  <div class="dp-action-desc" style="line-height:1.6;">{vreport.data_quality_certificate}</div>
  <div class="dp-cert-generated">Generated {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</div>
</div>""", unsafe_allow_html=True)
    st.write("")

# ── Transformation log (HTML table to avoid white-box issue) ───────────────────
if tlog and tlog.records:
    with st.expander("Transformation Log", expanded=False):
        log_data = pd.DataFrame([
            {
                "Action": r.action_type,
                "Column": r.column_name or "—",
                "Rows Affected": r.rows_affected,
                "Status": "✓ OK" if r.success else f"✗ {r.error_message or ''}",
            }
            for r in tlog.records
        ])
        st.markdown(render_data_preview(log_data, max_rows=50), unsafe_allow_html=True)

# ── Data preview (HTML table) ──────────────────────────────────────────────────
st.divider()
st.markdown(render_section_title("Data Preview"), unsafe_allow_html=True)
st.markdown(render_data_preview(cleaned, max_rows=20), unsafe_allow_html=True)

st.divider()

# ── Downloads ─────────────────────────────────────────────────────────────────
st.markdown(render_section_title("Download Cleaned Data"), unsafe_allow_html=True)
fname = "cleaned_data"

dl1, dl2, dl3 = st.columns(3)
with dl1:
    st.markdown('<div class="dp-card dp-card-center">', unsafe_allow_html=True)
    st.markdown("📊 **CSV**\nComma-separated values, universal compatibility")
    csv_bytes = cleaned.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv_bytes, f"{fname}.csv", "text/csv", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
with dl2:
    st.markdown('<div class="dp-card dp-card-center">', unsafe_allow_html=True)
    st.markdown("📗 **Excel**\nFor Microsoft Excel users")
    excel_buf = io.BytesIO()
    with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
        cleaned.to_excel(writer, index=False, sheet_name="Cleaned")
    st.download_button(
        "Download Excel", excel_buf.getvalue(), f"{fname}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)
with dl3:
    st.markdown('<div class="dp-card dp-card-center">', unsafe_allow_html=True)
    st.markdown("⚡ **Parquet**\nOptimized for ML pipelines and big data")
    parquet_buf = io.BytesIO()
    _pq = cleaned.copy()
    for _c in _pq.select_dtypes(include="object").columns:
        _pq[_c] = _pq[_c].astype(str)
    _pq.to_parquet(parquet_buf, index=False)
    st.download_button(
        "Download Parquet", parquet_buf.getvalue(), f"{fname}.parquet",
        "application/octet-stream",
        use_container_width=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ── Audit Log ─────────────────────────────────────────────────────────────────
st.divider()
with st.expander("Enterprise Audit Log", expanded=False):
    try:
        from src.governance.audit_log import load_audit_log
        events = load_audit_log()
        if events:
            audit_data = pd.DataFrame([
                {
                    "Timestamp": e.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "File": e.file_name,
                    "Format": e.file_format.upper(),
                    "Quality Before": f"{e.quality_score_before:.0f}",
                    "Quality After": f"{e.quality_score_after:.0f}",
                    "Improvement": f"+{e.improvement_pct:.1f}",
                    "Actions": f"{e.approved_actions}/{e.total_actions}",
                    "Rows Modified": e.rows_modified,
                    "Backend": e.backend_used,
                }
                for e in events[:10]
            ])
            st.markdown(render_data_preview(audit_data, max_rows=10), unsafe_allow_html=True)
            st.caption(f"Showing last {len(audit_data)} runs · stored in audit.jsonl")
        else:
            st.info("No audit events yet.")
    except Exception as _e:
        st.warning(f"Could not load audit log: {_e}")

st.divider()

# ── Feature Engineering Checkpoint ────────────────────────────────────────────
st.markdown(f"""
<div class="dp-card">
  <h2 class="dp-section-title">⚙ Make Your Data ML-Ready</h2>
  <p class="dp-section-subtitle">
    Your data is clean (quality score: <strong>{score_after:.0f}/100</strong>).
    The <strong>Feature Engineering Agent</strong> can now analyze your cleaned dataset
    and recommend encoding, scaling, and feature creation transformations to prepare it
    for machine learning.
  </p>
</div>""", unsafe_allow_html=True)

fe_target = st.text_input(
    "Target column (optional)",
    placeholder="e.g. price, churn — leave blank for general ML prep",
    key="fe_target_input",
)

st.markdown('<div class="dp-next-btn-container">', unsafe_allow_html=True)
if st.button("Continue to Feature Engineering →", type="primary", use_container_width=True):
    st.session_state.fe_target_column = fe_target.strip() or None
    st.switch_page("pages/5_Feature_Engineering.py")
st.markdown('</div>', unsafe_allow_html=True)

st.divider()
if st.button("Start Over", use_container_width=True):
    keys = ["uploaded_file", "raw_df", "file_metadata", "profile_report",
            "cleaning_plan", "cleaned_df", "validation_report", "transformation_log",
            "fe_plan", "ml_ready_df", "fe_log", "fe_target_column"]
    for k in keys:
        st.session_state[k] = None
    st.session_state.current_step = 1
    st.rerun()

st.markdown(render_footer(), unsafe_allow_html=True)
