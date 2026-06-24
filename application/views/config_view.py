"""Config page: pick the methods to compare and every ExperimentConfig field."""
from __future__ import annotations

import numpy as np
import streamlit as st

from core.experiment import METHOD_CATALOG, VALID_VARIANTS
from .components import section, card, parse_float_list, parse_int_list


def _method_picker(n_arms: int):
    """Checkbox + per-method variant selector. Returns the list of {name, variants} specs chosen."""
    section("Methods to compare", "Tick the methods; choose Capped/Uncapped per method (Kallus is parametric → flat).")
    chosen = []
    cols = st.columns(2)
    for i, m in enumerate(METHOD_CATALOG):
        col = cols[i % 2]
        with col:
            tag = []
            if m["wasserstein"]:
                tag.append("W")
            if m["maximize"]:
                tag.append("max-flag")
            suffix = f"  ·  {', '.join(tag)}" if tag else ""
            include = st.checkbox(f"**{m['name']}**{suffix}", value=m["name"] in ("IPW-O-W", "IPW-O-X", "IPW-X-X", "Kallus"),
                                  key=f"use_{m['name']}")
            if not include:
                continue
            if m["parametric"]:
                st.caption("Flat (parametric softmax — no capacity)")
                chosen.append({"name": m["name"], "variants": ["Flat"]})
            else:
                variants = st.multiselect("variants", list(VALID_VARIANTS), default=list(VALID_VARIANTS),
                                          key=f"var_{m['name']}", label_visibility="collapsed")
                if variants:
                    chosen.append({"name": m["name"], "variants": variants})
    if not chosen:
        st.warning("Select at least one method.")
    return chosen


def _capacity_editor(n_arms: int, prev):
    section("Capacity", "Per-arm budget (1/n)Σπ_k ≤ capₖ for the Capped variants. Arm 0 (control) is the "
            "uncapped fallback; keep cap₀ = 1. The caps must sum to ≥ 1 to stay feasible.")
    default = list(prev) if prev is not None and len(prev) == n_arms else [1.0] + [round(1.0 / n_arms + 0.1, 2)] * (n_arms - 1)
    cols = st.columns(n_arms)
    return [float(cols[k].number_input(f"cap arm {k}", 0.0, 1.0, value=float(default[k]), step=0.05,
                                       key=f"cap_{k}")) for k in range(n_arms)]


def render_config_page():
    st.title("①  Configure the experiment")
    n_arms = int(st.session_state.get("dgp").n_arms) if st.session_state.get("dgp") else 2
    card(f"Number of arms <b>K = {n_arms}</b> is taken from the DGP page, so the config can't disagree with the "
         f"data. Change K on the <b>Design DGP</b> page.", accent="#0984e3")

    methods = _method_picker(n_arms)

    section("Sensitivity sweep (Γ)", "Γ = assumed strength of unobserved confounding. The matched Γ = e^{γ_true/2} "
            "is always added to the sweep.")
    c1, c2 = st.columns(2)
    gamma_true = float(c1.number_input("γ_true (DGP confounding strength)", 0.0, 10.0, 3.0, step=0.5))
    gammas_txt = c2.text_input("Γ grid (comma-separated)", value="1, 3, 7")

    section("Replication & sizes")
    c1, c2, c3 = st.columns(3)
    seeds_txt = c1.text_input("seeds", value="0, 1, 2")
    n_train = int(c2.number_input("n_train", 50, 5000, 300, step=50))
    n_test = int(c3.number_input("n_test", 50, 20000, 800, step=100))

    cap = _capacity_editor(n_arms, st.session_state.get("_cap_prev"))
    st.session_state["_cap_prev"] = cap

    section("Convention & geometry")
    c1, c2, c3 = st.columns(3)
    maximize = c1.toggle("maximize (Y is reward)", value=True,
                         help="ON: Y is a reward (the studies' convention). OFF: Y is a loss — the paper's convention.")
    zscore = c2.toggle("z-score Wasserstein cost", value=True,
                       help="Standardise covariates before the Wasserstein ground cost (R-OW / Hajek-OW).")
    metric = c3.selectbox("metric", ["euclidean"], index=0)

    section("Support (discretisation)", "Snap continuous X onto a grid so the free-π LP is well-posed.")
    c1, c2, c3, c4 = st.columns(4)
    snap = c1.toggle("discretize (snap to grid)", value=True)
    mesh = int(c2.number_input("mesh (levels/axis)", 2, 20, 6, step=1))
    lo = float(c3.number_input("mesh lo", value=-1.0, step=0.5))
    hi = float(c4.number_input("mesh hi", value=1.0, step=0.5))
    rounding = int(st.number_input("rounding_digits", 1, 10, 6, step=1))

    experiment = st.text_input("experiment name (results/<name>/)", value="policy-lab-run")

    # assemble + validate-lite, store for the Run page
    errors = []
    try:
        gammas = parse_float_list(gammas_txt, name="Γ grid")
    except ValueError as e:
        errors.append(str(e)); gammas = []
    try:
        seeds = parse_int_list(seeds_txt, name="seeds")
    except ValueError as e:
        errors.append(str(e)); seeds = []
    if not seeds:
        errors.append("Provide at least one seed.")
    if hi <= lo:
        errors.append("mesh hi must be greater than mesh lo.")

    st.session_state["config"] = dict(
        experiment=experiment, n_arms=n_arms, cap=cap, gamma_true=gamma_true, gammas=gammas, seeds=seeds,
        methods=methods, maximize=maximize, n_train=n_train, n_test=n_test,
        zscore=zscore, metric=metric, snap=snap, mesh=mesh, mesh_range=(lo, hi), rounding_digits=rounding,
    )
    st.session_state["config_errors"] = errors
    if errors:
        for e in errors:
            st.error(e)
    else:
        st.success(f"Config ready: {len(methods)} method(s), {len(gammas)+1} Γ point(s), {len(seeds)} seed(s). "
                   f"Go to **③ Run**.")
