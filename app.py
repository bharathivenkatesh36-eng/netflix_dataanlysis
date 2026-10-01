"""
app.py
======
Entry point / router for the Netflix Content Intelligence Dashboard.

This project's Home page lives at `pages/1_Home.py` (so it appears in its
natural place in the sidebar's page order). `app.py` itself just bootstraps
straight to it, so `streamlit run app.py` lands the user on Home immediately
either way.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Netflix Content Intelligence Dashboard",
    page_icon=":material/movie:",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    st.switch_page("pages/1_Home.py")
except Exception:
    # Fallback for older Streamlit versions without st.switch_page: show a
    # plain link instead of a hard redirect.
    st.title("Netflix Content Intelligence Dashboard")
    st.markdown("### Please open **Home** from the sidebar to get started.")
    st.page_link("pages/1_Home.py", label="Go to Home", icon=":material/home:")
