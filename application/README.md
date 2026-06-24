# Confounding-Robust Policy Lab (GUI)

A Streamlit app for **policy optimization under unobserved confounding**: design a (confounded) DGP,
pick the methods to compare and every experiment setting, run, and inspect the results — all on top of
the `code 1.1` engine (`common/`, the method solvers, `run_experiment.py`).

## Run

```bash
# from the code 1.1/ root
pip install -r application/requirements.txt
streamlit run application/app.py
```

It opens in the browser. A run flows **① Configure → ② Design DGP → ③ Run → ④ Results**.
The number of arms **K** is set on the DGP page and shared with the config so they can never disagree.

> Running an experiment needs **Gurobi** (the LP backend) with a licence (academic licences are free).
> Without it you can still design and preview a DGP; only the *Run* step requires the solver.

## Pages

| Page | What it does |
|------|--------------|
| **① Configure** | Select methods to compare (Capped/Uncapped per method; Kallus is parametric/flat) and every `ExperimentConfig` field: capacity, Γ sweep, seeds, sizes, the `maximize` convention, the Wasserstein geometry (z-score / metric), and the support discretisation (snap / mesh / range / rounding). |
| **② Design DGP** | Choose #treatments K, the **X** and **S** distributions, and the outcome model `P(Y=1\|X,S,T=k)=σ(aₖ·X+bₖ·S+cₖ)` and confounding model `P(T=k\|X,S)=softmax(vₖ·X+γ·dₖ·(S−E[S]))` via per-arm coefficient tables — or write a custom `generate(n, gamma, rng)` in the **Advanced (Python)** tab. Live sample preview. |
| **③ Run** | Reviews the assembled config + DGP and runs `run_experiment`, saving the full tree under `results/<name>/`. |
| **④ Results** | Method-comparison plot (deployed test value vs Γ) with Full-info / Best-means ceilings, per-arm usage, an overlap-diagnostics banner, and a final-numbers table. |

## Architecture (clean separation)

```
application/
  app.py                 # entrypoint: theme + header + sidebar nav + session bootstrap
  .streamlit/config.toml # theme
  requirements.txt
  core/                  # PURE LOGIC — no Streamlit, unit-tested in ../tests/test_application.py
    dgp.py               #   DGPSpec + structured/code → generate(n, gamma, rng); preview; defaults
    experiment.py        #   UI state → ExperimentConfig; run (captures stdout); load_results
    plotting.py          #   matplotlib figures (comparison, usage, DGP preview)
  views/                 # STREAMLIT — one render_* per page + shared components
    config_view.py  dgp_view.py  run_view.py  results_view.py  components.py
```

`core/` never imports Streamlit, so it is testable headless (`pytest tests/test_application.py`). The
views hold all UI; `app.py` only wires them together. S is the **unobserved** confounder — it drives both
the outcome and the treatment, but no method ever sees it (the methods only ever estimate `P(T|X)` from data).
