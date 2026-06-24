"""Confounding-Robust Policy Lab — Streamlit entrypoint.

Run with:   streamlit run application/app.py     (from the code-1.1 root)

Thin entrypoint: theme + header + sidebar navigation + session bootstrap, delegating each page to a
``render_*`` function in ``views/``. All non-UI logic lives in ``core/`` (importable + tested separately).
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# make 'core' and 'views' importable no matter where streamlit is launched from
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from views.config_view import render_config_page          # noqa: E402
from views.dgp_view import render_dgp_page                # noqa: E402
from views.run_view import render_run_page                # noqa: E402
from views.results_view import render_results_page        # noqa: E402

st.set_page_config(page_title="Confounding-Robust Policy Lab", page_icon="⚙️", layout="wide")

# ----------------------------------------------------------------- modern styling
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.2rem; max-width: 1100px; }
      h1 { font-weight: 750; letter-spacing: -0.5px; }
      h3 { margin-top: 1.1rem; }
      .stButton>button { border-radius: 10px; font-weight: 600; padding: 0.5rem 1.1rem; }
      div[data-testid="stMetric"] { background:#f7f7fc; border:1px solid #ececf6; border-radius:12px; padding:10px 14px; }
      section[data-testid="stSidebar"] { background:#11122a; }
      section[data-testid="stSidebar"] * { color:#e9e9f5 !important; }
      section[data-testid="stSidebar"] .stRadio label { font-size: 1.02rem; }
      .lab-hero { background:linear-gradient(110deg,#6c5ce7 0%,#341f97 60%,#0c0c2b 100%);
                  color:#fff; border-radius:16px; padding:20px 26px; margin-bottom:14px; }
      .lab-hero h2 { color:#fff; margin:0; font-weight:760; letter-spacing:-0.5px; }
      .lab-hero p  { color:#d9d6ff; margin:.3rem 0 0; font-size:.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="lab-hero">
      <h2>⚙️ Confounding-Robust Policy Lab</h2>
      <p>Design a confounded DGP, pick the methods to compare, and evaluate policy learning under
         unobserved confounding — R-OW · Hajek-OW · DoublyRobust · Kallus · IPW · Oracle and more.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

PAGES = {
    "①  Configure": render_config_page,
    "②  Design DGP": render_dgp_page,
    "③  Run": render_run_page,
    "④  Results": render_results_page,
}

with st.sidebar:
    st.markdown("## ⚙️ Policy Lab")
    page = st.radio("Navigate", list(PAGES), label_visibility="collapsed", key="nav")
    st.divider()
    st.caption("A run flows ① → ② → ③ → ④.\nK (number of arms) is set on the DGP page and shared with the config.")
    st.caption("code 1.1 · application")

PAGES[page]()
