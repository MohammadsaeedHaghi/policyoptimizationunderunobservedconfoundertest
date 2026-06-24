"""Small shared Streamlit UI helpers (section headers, info cards, list parsers)."""
from __future__ import annotations

from typing import List

import streamlit as st


def section(title: str, subtitle: str = "") -> None:
    """A consistent section header with an optional muted subtitle."""
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def card(body_md: str, *, accent: str = "#6c5ce7") -> None:
    """A left-accented info card (matches the project's themed-card aesthetic)."""
    st.markdown(
        f"<div style='border:1px solid #e6e6ef;border-left:4px solid {accent};border-radius:10px;"
        f"padding:12px 16px;margin:8px 0;background:#fff'>{body_md}</div>",
        unsafe_allow_html=True,
    )


def parse_float_list(text: str, *, name: str = "values") -> List[float]:
    """Parse a comma/space separated list of floats, raising a friendly error on bad input."""
    items = [t for t in text.replace(",", " ").split() if t]
    try:
        return [float(t) for t in items]
    except ValueError as e:
        raise ValueError(f"Could not parse {name} from {text!r}: {e}") from e


def parse_int_list(text: str, *, name: str = "values") -> List[int]:
    items = [t for t in text.replace(",", " ").split() if t]
    try:
        return [int(float(t)) for t in items]
    except ValueError as e:
        raise ValueError(f"Could not parse {name} from {text!r}: {e}") from e
