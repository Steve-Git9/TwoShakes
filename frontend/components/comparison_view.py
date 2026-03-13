"""Before/after data comparison component."""
import streamlit as st
import pandas as pd


def render_comparison(df_before: pd.DataFrame, df_after: pd.DataFrame, n: int = 10):
    """Side-by-side before/after preview."""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        st.dataframe(df_before.head(n), use_container_width=True)
    with col2:
        st.subheader("After")
        st.dataframe(df_after.head(n), use_container_width=True)
