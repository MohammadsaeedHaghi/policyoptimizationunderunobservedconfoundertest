"""DGP page: design the data-generating process (structured forms + advanced Python) with a live preview."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.dgp import (DGPSpec, DistSpec, OutcomeModel, TreatmentModel,
                      default_spec, sample_preview, _CODE_TEMPLATE)
from core.presets import PRESETS, PRESET_NOTES, CUSTOM
from core.plotting import dgp_preview_figure
from .components import section, card

_X_DISTS = ("uniform", "normal")
_S_DISTS = ("bernoulli", "uniform", "normal")


def _dist_editor(label: str, options, current: DistSpec, key: str, *, allow_dim=False) -> DistSpec:
    """Render a distribution selector + its parameters; return the chosen DistSpec."""
    cols = st.columns([1.2, 1, 1, 1])
    name = cols[0].selectbox(label, options, index=options.index(current.name) if current.name in options else 0,
                             key=f"{key}_name")
    p = dict(current.params)
    if name == "uniform":
        p["low"] = cols[1].number_input("low", value=float(current.params.get("low", -1.0)), key=f"{key}_lo")
        p["high"] = cols[2].number_input("high", value=float(current.params.get("high", 1.0)), key=f"{key}_hi")
    elif name == "normal":
        p["mean"] = cols[1].number_input("mean", value=float(current.params.get("mean", 0.0)), key=f"{key}_mu")
        p["std"] = cols[2].number_input("std", value=float(current.params.get("std", 1.0)), min_value=1e-6, key=f"{key}_sd")
    elif name == "bernoulli":
        p["p"] = cols[1].number_input("p", value=float(current.params.get("p", 0.5)), min_value=0.0, max_value=1.0, key=f"{key}_p")
    dim = current.dim
    if allow_dim:
        dim = int(cols[3].number_input("dim d", 1, 4, value=int(current.dim), step=1, key=f"{key}_dim"))
    return DistSpec(name, p, dim=dim)


def _coef_table(title, columns, values, *, key, help_md=""):
    """A per-arm coefficient editor (rows = arms). Returns the edited (n_arms × n_cols) array."""
    st.markdown(f"**{title}**")
    if help_md:
        st.caption(help_md)
    df = pd.DataFrame(values, columns=columns, index=[f"arm {k}" for k in range(len(values))])
    edited = st.data_editor(df, key=key, width="stretch")
    return edited.to_numpy(dtype=float)


def _structured_editor(spec: DGPSpec) -> DGPSpec:
    c1, c2 = st.columns(2)
    K = int(c1.number_input("Number of treatments  K", 2, 6, value=spec.n_arms, step=1,
                            help="arm 0 is the control / never-capped fallback"))
    d = int(c2.number_input("Covariate dimension  d", 1, 4, value=spec.x.dim, step=1))
    if K != spec.n_arms or d != spec.x.dim:                       # dims changed → reset to sensible defaults
        spec = default_spec(K, d)
        st.info(f"Reset coefficients to defaults for K={K}, d={d}.")

    section("Distributions", "X is observed; S is the UNOBSERVED confounder that drives both outcome and treatment.")
    x = _dist_editor("X  (covariates)", _X_DISTS, DistSpec(spec.x.name, spec.x.params, d), key="xd")
    x.dim = d
    s = _dist_editor("S  (unobserved confounder, scalar)", _S_DISTS, spec.s, key="sd")

    section("Outcome model", "P(Y=1 | X, S, T=k) = σ( aₖ·X + bₖ·S + cₖ ).  Edit the per-arm coefficients.")
    out_cols = [f"a·x{j}" for j in range(d)] + ["b·S", "c"]
    out_vals = np.column_stack([np.asarray(spec.outcome.a, float).reshape(K, d),
                                np.asarray(spec.outcome.b, float).reshape(K, 1),
                                np.asarray(spec.outcome.c, float).reshape(K, 1)])
    out = _coef_table("σ( aₖ·X + bₖ·S + cₖ )", out_cols, out_vals, key=f"out_{K}_{d}")
    a, b, c = out[:, :d].tolist(), out[:, d].tolist(), out[:, d + 1].tolist()

    section("Treatment (confounding) model",
            "P(T=k | X, S) = softmaxₖ( vₖ·X + γ·dₖ·(S − E[S]) ).  γ is the run's confounding strength.")
    tr_cols = [f"v·x{j}" for j in range(d)] + ["d·S (×γ)"]
    tr_vals = np.column_stack([np.asarray(spec.treatment.v, float).reshape(K, d),
                               np.asarray(spec.treatment.d_s, float).reshape(K, 1)])
    tr = _coef_table("softmaxₖ( vₖ·X + γ·dₖ·(S−E[S]) )", tr_cols, tr_vals, key=f"tr_{K}_{d}",
                     help_md="Keep arm 0 (control) unconfounded (d·S = 0) so it stays a clean fallback.")
    v, d_s = tr[:, :d].tolist(), tr[:, d].tolist()

    return DGPSpec(n_arms=K, x=x, s=s, outcome=OutcomeModel(a, b, c),
                   treatment=TreatmentModel(v, d_s), mode="structured")


def _code_editor(spec: DGPSpec) -> DGPSpec:
    card("Write a <code>generate(n, gamma, rng)</code> that returns <code>(X, T, Y, Ypot, mu)</code> "
         "or a <code>SimpleNamespace</code> with those fields. <code>numpy</code> is available as "
         "<code>np</code>. Shapes are validated when you preview/run.", accent="#e17055")
    code = st.text_area("generate(n, gamma, rng)", value=spec.code or _CODE_TEMPLATE, height=340,
                        key="dgp_code", label_visibility="collapsed")
    return DGPSpec(n_arms=spec.n_arms, x=spec.x, s=spec.s, outcome=spec.outcome,
                   treatment=spec.treatment, mode="code", code=code)


def _preview(spec: DGPSpec) -> None:
    section("Live preview", "A single sample draw at the chosen settings.")
    c1, c2, c3 = st.columns([1, 1, 1])
    n = int(c1.number_input("preview n", 100, 5000, 400, step=100, key="prev_n"))
    gamma = float(c2.number_input("preview γ", 1.0, 10.0, 3.0, step=0.5, key="prev_g"))
    seed = int(c3.number_input("preview seed", 0, 9999, 0, step=1, key="prev_seed"))
    if st.button("🔄  Preview sample draw", type="primary"):
        try:
            pv = sample_preview(spec, n=n, gamma=gamma, seed=seed)
        except Exception as e:                                    # surface DGP/code errors cleanly
            st.error(f"DGP error: {e}")
            return
        m1, m2, m3 = st.columns(3)
        m1.metric("arms covered", f"{sum(c>0 for c in pv['arm_counts'])}/{pv['n_arms']}")
        m2.metric("outcome rate  E[Y]", f"{pv['y_rate']:.3f}")
        m3.metric("arm counts", " · ".join(str(c) for c in pv["arm_counts"]))
        if not pv["all_arms_present"]:
            st.warning("Some arm has 0 observations at this draw — the LP solvers need ≥1 obs per arm. "
                       "Increase n, soften the treatment model, or check the code.")
        st.pyplot(dgp_preview_figure(pv), width="stretch")


def render_dgp_page():
    st.title("②  Design the DGP")
    st.caption("Load a study preset or build your own: the number of treatments, the X / S distributions, "
               "and the outcome & confounding models — or write a custom generator. S is unobserved "
               "(it confounds both T and Y).")
    st.session_state.setdefault("dgp", default_spec(2, 2))

    # ---- preset loader: the study DGPs where R-OW (Wasserstein) wins ----
    preset = st.selectbox("📦  Load a preset DGP", list(PRESETS), key="preset_select",
                          help="Study DGPs ported verbatim from the discrete-X experiments; selecting one "
                               "loads its exact generator (Advanced/Python mode) so a run reproduces the study.")
    if preset == CUSTOM:
        st.session_state["_preset_active"] = None
    elif st.session_state.get("_preset_active") != preset:        # a (new) preset was chosen → load it
        st.session_state.dgp = PRESETS[preset]                    # exact code-mode DGPSpec
        st.session_state["_preset_active"] = preset
        st.session_state["dgp_mode"] = "Advanced (Python)"        # presets are code mode
        st.session_state.pop("dgp_code", None)                    # force the code editor to reseed
        st.rerun()
    if preset != CUSTOM:
        card(PRESET_NOTES.get(preset, ""), accent="#00b894")
    st.divider()

    spec = st.session_state.dgp
    mode = st.radio("Authoring mode", ["Structured", "Advanced (Python)"], horizontal=True, key="dgp_mode")
    spec = _structured_editor(spec) if mode == "Structured" else _code_editor(spec)
    st.session_state.dgp = spec
    st.divider()
    _preview(spec)
