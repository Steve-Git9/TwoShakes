"""Styled file upload zone component."""
import streamlit as st


def render_upload_zone() -> object:
    """Render styled upload area; returns the uploaded file object or None."""
    badges = ["CSV", "TSV", "XLSX", "XLS", "JSON", "XML"]
    html = " ".join(
        f'<span class="format-badge">{b}</span>' for b in badges
    )
    st.markdown(html, unsafe_allow_html=True)
    st.write("")
    return st.file_uploader(
        "Drop your file here or click to browse",
        type=["csv", "tsv", "xlsx", "xls", "json", "xml"],
        label_visibility="collapsed",
    )
