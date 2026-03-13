"""Two Shakes DataPrepAgent — Streamlit multi-page app entry point."""
import sys, os, pathlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st

st.set_page_config(
    page_title="Two Shakes",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.ui_helpers import load_css, render_sidebar, render_metric_card, render_hero, render_footer

load_css()

# ── Session state defaults ─────────────────────────────────────────────────────
_defaults = dict(
    uploaded_file=None,
    raw_df=None,
    file_metadata=None,
    profile_report=None,
    cleaning_plan=None,
    cleaned_df=None,
    validation_report=None,
    transformation_log=None,
    current_step=1,
    fe_plan=None,
    ml_ready_df=None,
    fe_log=None,
    fe_target_column=None,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────
render_sidebar()

# ── Hero: logo + subtitle + feature pills ─────────────────────────────────────
_LOGO = pathlib.Path(__file__).parent / "static" / "two shakes.png"
_, logo_col, _ = st.columns([1.75, 0.5, 1.75])
with logo_col:
    if _LOGO.exists():
        st.image(str(_LOGO), use_container_width=True)

st.markdown(render_hero(
    "",
    "Transform messy data into analysis-ready datasets",
    ["Multi-format support", "AI-powered cleaning", "ML-ready output", "Enterprise audit log"],
), unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_metric_card("Step 1", "Upload", icon="📄"), unsafe_allow_html=True)
with c2:
    st.markdown(render_metric_card("Step 2", "Profile", icon="🔍"), unsafe_allow_html=True)
with c3:
    st.markdown(render_metric_card("Step 3", "Review", icon="🧹"), unsafe_allow_html=True)
with c4:
    st.markdown(render_metric_card("Step 4", "Download", icon="✅"), unsafe_allow_html=True)

st.divider()

st.markdown('<div class="dp-next-btn-container">', unsafe_allow_html=True)
if st.button("Get Started: Upload Your Data →", use_container_width=True):
    st.switch_page("pages/1_Upload.py")
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(render_footer(), unsafe_allow_html=True)
