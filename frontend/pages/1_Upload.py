"""Upload & Ingest page."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import asyncio
import concurrent.futures
import streamlit as st
import pandas as pd

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


st.markdown(render_section_title("Upload Your Data", "Drop a file and let the AI do the rest."), unsafe_allow_html=True)

# Format badges
badges = ["CSV", "TSV", "XLSX", "XLS", "JSON", "XML", "PDF"]
html = " ".join(f'<span class="dp-format-badge">{b}</span>' for b in badges)
st.markdown(html, unsafe_allow_html=True)
st.write("")

# File uploader
uploaded = st.file_uploader(
    "Drag & drop your file here, or click to browse",
    type=["csv", "tsv", "xlsx", "xls", "json", "xml", "pdf"],
    help="Supported: CSV, TSV, Excel, JSON, XML, PDF (via Azure Document Intelligence)",
)

if uploaded:
    st.session_state.uploaded_file = uploaded

    size_kb = len(uploaded.getvalue()) / 1024
    size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb/1024:.2f} MB"
    ext = uploaded.name.rsplit(".", 1)[-1].upper()

    st.markdown(f"""
<div class="dp-card">
  <div style="display:flex;gap:2rem;align-items:center;flex-wrap:wrap;">
    <div style="font-size:2.5rem;">📄</div>
    <div>
      <div style="font-weight:700;font-size:1rem;color:var(--text-1);">{uploaded.name}</div>
      <div class="dp-action-desc" style="margin-top:2px;">{size_str} · <span class="dp-format-badge">{ext}</span></div>
    </div>
  </div>
</div>""", unsafe_allow_html=True)

    # Raw preview if already analyzed
    if st.session_state.raw_df is not None:
        with st.expander("Raw Data Preview (first 5 rows)", expanded=False):
            st.dataframe(st.session_state.raw_df.head(5), use_container_width=True)

    st.write("")
    if st.session_state.get("profile_report") is not None:
        # ── Analysis already done: show result card + prominent CTA ──────────
        profile = st.session_state.profile_report
        meta_s  = st.session_state.file_metadata
        st.markdown(f"""
<div class="dp-cert-box">
  <div class="dp-cert-title">✅ Analysis Complete</div>
  <p class="dp-section-subtitle" style="margin:0.3rem 0 0 0;">
    <strong>{meta_s.row_count:,} rows × {meta_s.col_count} columns</strong> profiled
    &nbsp;·&nbsp; Quality score: <strong>{profile.overall_quality_score:.0f}/100</strong>
    &nbsp;·&nbsp; <strong>{len(st.session_state.cleaning_plan.actions)}</strong> cleaning actions ready
  </p>
</div>""", unsafe_allow_html=True)
        st.markdown('<div class="dp-next-btn-container">', unsafe_allow_html=True)
        if st.button("View Data Profile →", type="primary", use_container_width=True):
            st.switch_page("pages/2_Profile.py")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # ── Not yet analyzed: show the Analyze button ─────────────────────────
        if st.button("Analyze Data →", type="primary", use_container_width=True):
            import tempfile
            tmp = tempfile.NamedTemporaryFile(
                delete=False,
                suffix="." + uploaded.name.rsplit(".", 1)[-1],
            )
            tmp.write(uploaded.getvalue())
            tmp.close()

            status = st.status("Analyzing your data…", expanded=True)
            with status:
                st.write("📄 Parsing file…")
                from src.agents.ingestion_agent import ingest
                df, meta = _run(ingest(tmp.name))
                st.session_state.raw_df = df
                st.session_state.file_metadata = meta
                st.session_state.current_step = max(st.session_state.current_step, 2)

                st.write(f"✅ Parsed {meta.row_count:,} rows × {meta.col_count} columns")
                st.write("🔍 Running AI profiler…")

                from src.agents.profiler_agent import profile_dataframe
                profile = _run(profile_dataframe(df, meta))
                st.session_state.profile_report = profile

                st.write(f"✅ Quality score: {profile.overall_quality_score:.0f}/100")
                st.write("🤖 Generating cleaning plan…")

                from src.agents.strategy_agent import generate_cleaning_plan
                plan = _run(generate_cleaning_plan(profile, df))
                st.session_state.cleaning_plan = plan
                st.session_state.current_step = max(st.session_state.current_step, 3)

                st.write(f"✅ {len(plan.actions)} cleaning actions ready")
                status.update(label="Analysis complete!", state="complete", expanded=False)

            os.unlink(tmp.name)
            st.rerun()

else:
    st.markdown(render_empty_state(
        "📂",
        "No file uploaded yet",
        "Drop a CSV, Excel, JSON, XML, or PDF file above to get started.",
    ), unsafe_allow_html=True)

if st.session_state.raw_df is not None and uploaded is None:
    st.markdown(render_section_title("Previously Loaded Data"), unsafe_allow_html=True)
    df = st.session_state.raw_df
    meta = st.session_state.file_metadata
    if meta:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{meta.row_count:,}")
        c2.metric("Columns", meta.col_count)
        c3.metric("Format", meta.file_format.upper() if meta.file_format else "?")
    st.dataframe(df.head(10), use_container_width=True)

st.markdown(render_footer(), unsafe_allow_html=True)
