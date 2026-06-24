"""Results page: comparison plot, usage, diagnostics banner, and a final-numbers table."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from core.plotting import comparison_figure, usage_figure
from .components import section


def render_results_page():
    st.title("④  Results")
    res = st.session_state.get("results")
    if not res:
        st.info("Run an experiment on the **③  Run** page first.")
        return
    st.caption(f"results dir: `{res['exp_dir']}`")

    # ---- diagnostics banner (overlap on the estimated weights) ----
    diags = res.get("diagnostics", [])
    if diags:
        bad = [d for d in diags if not d.get("overlap_ok", True)]
        if bad:
            st.warning(f"⚠ {len(bad)}/{len(diags)} seed(s) have poor overlap "
                       f"(min propensity {min(d['min_propensity'] for d in bad):.3g}) — "
                       f"the inverse weights may be driving the results.")
        else:
            st.success(f"Overlap OK on all {len(diags)} seed(s) "
                       f"(min propensity {min(d['min_propensity'] for d in diags):.3g}).")

    # ---- comparison plot ----
    section("Method comparison", "Deployed value on the held-out test set vs Γ. Higher is better.")
    field = st.radio("outcome", ["realized_test", "exp_test"], horizontal=True,
                     format_func=lambda f: {"realized_test": "realised test  E[Y]",
                                            "exp_test": "true-mean test  E[μ]"}[f])
    st.pyplot(comparison_figure(res, field=field), width="stretch")

    section("Policy usage", "Per-arm realised usage at the matched Γ.")
    st.pyplot(usage_figure(res), width="stretch")

    # ---- final-numbers table ----
    section("Final numbers", "Deployed test value (mean over seeds) at the matched Γ.")
    rows = []
    for label, mrows in sorted(res["methods"].items()):
        if not mrows:
            continue
        gmax = max(round(float(r["Gamma"]), 4) for r in mrows)
        sel = [r for r in mrows if round(float(r["Gamma"]), 4) == gmax]
        rows.append({
            "method": label,
            "realised test E[Y]": float(np.mean([r["realized_test"] for r in sel])),
            "true-mean test E[μ]": float(np.mean([r["exp_test"] for r in sel])),
            "objective": float(np.mean([r["objective_value"] for r in sel])),
            "obj_kind": sel[0].get("obj_kind", ""),
        })
    if rows:
        df = pd.DataFrame(rows).set_index("method")
        st.dataframe(df.style.format({c: "{:.4f}" for c in df.columns if c != "obj_kind"}),
                     width="stretch")
    if res.get("ceilings"):
        st.caption("Ceilings — " + " · ".join(f"{k}: {v:.4f}" for k, v in res["ceilings"].items() if v is not None))

    with st.expander("config used (config.json)"):
        st.json(res["config"])
