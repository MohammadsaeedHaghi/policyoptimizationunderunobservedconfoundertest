"""Run page: review the assembled config + DGP, then execute the experiment."""
from __future__ import annotations

import streamlit as st

from core.dgp import build_generate
from core.experiment import build_experiment_config, run, load_results
from .components import section


def render_run_page():
    st.title("③  Run the experiment")
    cfg_state = st.session_state.get("config")
    dgp = st.session_state.get("dgp")
    errors = st.session_state.get("config_errors", [])

    if cfg_state is None:
        st.info("Set up the experiment on the **①  Configure** page first.")
        return
    if dgp is None:
        st.info("Design a DGP on the **②  Design DGP** page first.")
        return

    section("Review")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Methods**")
        for m in cfg_state["methods"]:
            st.write(f"• {m['name']}  ({', '.join(m['variants'])})")
        st.markdown(f"**Γ_true** `{cfg_state['gamma_true']}`  ·  **Γ grid** `{cfg_state['gammas']}`")
        st.markdown(f"**seeds** `{cfg_state['seeds']}`  ·  **n_train / n_test** "
                    f"`{cfg_state['n_train']} / {cfg_state['n_test']}`")
    with c2:
        st.markdown(f"**K** `{cfg_state['n_arms']}`  ·  **cap** `{cfg_state['cap']}`")
        st.markdown(f"**maximize** `{cfg_state['maximize']}`  ·  **z-score** `{cfg_state['zscore']}`  ·  "
                    f"**discretize** `{cfg_state['snap']}` (mesh `{cfg_state['mesh']}`)")
        st.markdown(f"**DGP** mode=`{dgp.mode}`, K=`{dgp.n_arms}`, "
                    f"X~`{dgp.x.name}({dgp.x.dim}d)`, S~`{dgp.s.name}`")

    if errors:
        st.error("Fix these on the ① page before running:")
        for e in errors:
            st.write("• ", e)
        return
    if not cfg_state["methods"]:
        st.error("Select at least one method on the ① page.")
        return

    clean = st.toggle("clean stale results for this experiment name", value=True,
                      help="Prune outputs left by a previous run of the same experiment name with a different config.")

    if st.button("▶  Run experiment", type="primary"):
        try:
            config = build_experiment_config(cfg_state)
            generate = build_generate(dgp)
        except Exception as e:
            st.error(f"Could not build the experiment: {e}")
            return
        with st.spinner("Running — fitting every method × variant × Γ × seed, then deploying to the test set…"):
            try:
                exp_dir, log = run(config, generate, clean=clean)
            except Exception as e:
                st.error(f"Run failed: {e}")
                st.exception(e)
                return
        st.session_state["results"] = load_results(exp_dir)
        st.success(f"Done → `{exp_dir}`")
        with st.expander("runner log"):
            st.code(log or "(no output)")
        st.info("Open the **④  Results** page to see the comparison.")
