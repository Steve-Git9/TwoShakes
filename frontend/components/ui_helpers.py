"""
UI helper functions for the Two Shakes DataPrep app.
All render_* functions return HTML strings for st.markdown(unsafe_allow_html=True).
render_sidebar() and load_css() call Streamlit APIs directly.
"""
from __future__ import annotations
from pathlib import Path
import streamlit as st

_LOGO_PATH = Path(__file__).parent.parent / "static" / "two shakes.png"

_SIDEBAR_STEPS = [
    {"label": "Home",                "icon": "🏠"},
    {"label": "Upload",              "icon": "📄"},
    {"label": "Profile",             "icon": "🔍"},
    {"label": "Cleaning Plan",       "icon": "🧹"},
    {"label": "Results",             "icon": "✅"},
    {"label": "Feature Engineering", "icon": "⚙"},
]

_SESSION_DEFAULTS = dict(
    uploaded_file=None, raw_df=None, file_metadata=None,
    profile_report=None, cleaning_plan=None, cleaned_df=None,
    validation_report=None, transformation_log=None,
    current_step=1, fe_plan=None, ml_ready_df=None,
    fe_log=None, fe_target_column=None,
)


def load_css() -> None:
    """Inject Google Fonts and the master CSS — must be the very first Streamlit call."""
    st.markdown(
        '<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1'
        '&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap"'
        ' rel="stylesheet">',
        unsafe_allow_html=True,
    )
    css_path = Path(__file__).parent.parent / "static" / "style.css"
    if css_path.exists():
        with open(css_path, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar() -> None:
    """Render the unified branded sidebar on every page."""
    with st.sidebar:
        if _LOGO_PATH.exists():
            st.image(str(_LOGO_PATH), width=100)
        else:
            st.markdown(
                '<div class="dp-sidebar-brand">'
                '<span class="dp-sidebar-brand-name">Two Shakes</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        st.divider()
        cur = st.session_state.get("current_step", 1)
        st.markdown(render_step_indicator(_SIDEBAR_STEPS, cur), unsafe_allow_html=True)
        st.divider()
        if st.button("Start Over", use_container_width=True, key="sidebar_start_over"):
            for k, v in _SESSION_DEFAULTS.items():
                st.session_state[k] = v
            st.switch_page("app.py")


def render_metric_card(label: str, value: str, delta: str = None, icon: str = None) -> str:
    delta_html = ""
    if delta:
        cls = "positive" if not delta.startswith("-") else "negative"
        delta_html = f'<span class="dm-delta {cls}">{delta}</span>'
    icon_html = f'<span class="dp-metric-icon">{icon}</span>' if icon else ""
    return f"""
<div class="dp-metric-card">
  {icon_html}
  <span class="dm-value">{value}</span>
  <span class="dm-label">{label}</span>
  {delta_html}
</div>"""


def render_quality_gauge(score: float) -> str:
    if score >= 80:
        color = "#2E7D32"
        label = "EXCELLENT"
    elif score >= 60:
        color = "#E65100"
        label = "FAIR"
    elif score >= 40:
        color = "#F39C12"
        label = "POOR"
    else:
        color = "#C62828"
        label = "CRITICAL"
    pct = int(score)
    return f"""
<div class="dp-quality-score">
  <div class="dp-quality-circle" style="background:conic-gradient({color} {pct * 3.6}deg, #E8E0DC 0deg);">
    <div style="width:88px;height:88px;border-radius:50%;background:white;display:flex;align-items:center;justify-content:center;position:absolute;">
      <span style="font-family:'DM Serif Display',serif;font-size:1.6rem;font-weight:700;color:{color};">{pct}</span>
    </div>
  </div>
  <span class="dp-quality-label" style="color:{color};">{label}</span>
</div>"""


def render_quality_bar(score: float) -> str:
    """Branded progress bar for per-column quality scores."""
    cls = "dp-quality-bar-good" if score >= 70 else "dp-quality-bar-warn" if score >= 40 else "dp-quality-bar-bad"
    return (
        f'<div class="dp-quality-bar {cls}">'
        f'<div class="dp-quality-bar-fill" style="width:{score:.0f}%;"></div>'
        f'</div>'
        f'<small class="dp-action-desc">Quality: {score:.0f}/100</small>'
    )


def render_badge(text: str, variant: str = "info") -> str:
    return f'<span class="dp-badge-{variant}">{text}</span>'


def render_step_indicator(steps: list[dict], current_step: int) -> str:
    items = []
    for i, step in enumerate(steps):
        step_num = i + 1
        if step_num < current_step:
            cls = "completed"
            dot = "✓"
        elif step_num == current_step:
            cls = "active"
            dot = str(step_num)
        else:
            cls = "pending"
            dot = str(step_num)
        icon  = step.get("icon", "")
        label = step.get("label", "")
        items.append(
            f'<div class="dp-step {cls}">'
            f'  <div class="dp-step-dot">{dot}</div>'
            f'  <span class="dp-step-label">{icon} {label}</span>'
            f'</div>'
        )
    return '<div class="dp-step-indicator">' + "".join(items) + "</div>"


def render_action_card(label: str, action_type: str, column: str, description: str,
                       reason: str = "", priority: str = "medium", warning: str = "") -> str:
    badge_map = {"high": "dp-badge-high", "medium": "dp-badge-medium", "low": "dp-badge-low"}
    badge_cls = badge_map.get(priority.lower(), "dp-badge-info")
    priority_label = priority.upper()
    warning_html = f'<div class="dp-action-warning">⚠ {warning}</div>' if warning else ""
    return f"""
<div class="dp-action-card priority-{priority.lower()}">
  <div class="dp-action-header priority-{priority.lower()}">
    <span class="{badge_cls}">{priority_label}</span>
    <span class="dp-action-type">{action_type}</span>
    <span class="dp-action-col">{column}</span>
  </div>
  <div class="dp-action-desc">{description}</div>
  {('<div class="dp-action-reason">Reason: ' + reason + '</div>') if reason else ''}
  {warning_html}
</div>"""


def render_comparison_metric(label: str, before: str, after: str, delta: str = None) -> str:
    delta_html = ""
    if delta:
        cls = "up" if not str(delta).startswith("-") else "down"
        delta_html = f'<span class="delta {cls}">{delta}</span>'
    return f"""
<div class="dp-comparison-row">
  <div class="dp-comparison-label">{label}</div>
  <div class="dp-comparison">
    <span class="before">{before}</span>
    <span class="arrow">→</span>
    <span class="after">{after}</span>
    {delta_html}
  </div>
</div>"""


def render_section_title(title: str, subtitle: str = None) -> str:
    sub_html = f'<p class="dp-section-subtitle">{subtitle}</p>' if subtitle else ""
    return f'<h2 class="dp-section-title">{title}</h2>{sub_html}'


def render_empty_state(icon: str, message: str, sub_message: str = None) -> str:
    sub_html = f'<div class="sub">{sub_message}</div>' if sub_message else ""
    return f"""
<div class="dp-empty-state">
  <span class="icon">{icon}</span>
  <div class="message">{message}</div>
  {sub_html}
</div>"""


def render_data_preview(df, max_rows: int = 10) -> str:
    rows = df.head(max_rows)
    headers = "".join(f"<th>{col}</th>" for col in rows.columns)
    body_rows = ""
    for _, row in rows.iterrows():
        cells = "".join(f"<td>{val}</td>" for val in row.values)
        body_rows += f"<tr>{cells}</tr>"
    return f"""
<div class="dp-table-container">
  <table class="dp-table">
    <thead><tr>{headers}</tr></thead>
    <tbody>{body_rows}</tbody>
  </table>
</div>"""


def render_hero(title: str, subtitle: str, features: list[str] = None) -> str:
    pills = ""
    if features:
        pills = "".join(f'<span class="dp-feature-pill">{f}</span>' for f in features)
        pills = f'<div class="dp-hero-pills">{pills}</div>'
    title_html = f'<h1 class="dp-hero-title">{title}</h1>' if title else ""
    return f"""
<div class="dp-hero">
  {title_html}
  <p class="dp-hero-sub">{subtitle}</p>
  {pills}
</div>"""


def render_footer() -> str:
    return """
<div class="dp-footer-badge">
  Powered by <strong>Microsoft Foundry</strong> &amp; <strong>Agent Framework</strong>
</div>"""
