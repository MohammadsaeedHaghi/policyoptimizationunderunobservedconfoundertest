---
name: application_app
description: "The Streamlit GUI app (code 1.1/application/) — design a DGP, pick methods + config, run, view results."
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

**`code 1.1/application/`** (built 2026-06-11) — a **Streamlit** GUI front-end over the code-1.1 engine. Run with
`streamlit run application/app.py` from the code-1.1 root (`pip install -r application/requirements.txt`; needs
streamlit≥1.50 + the project's Gurobi for the Run step). Four pages, flow ①→②→③→④:
- **① Configure** — pick methods to compare (Capped/Uncapped per method; Kallus→Flat) + every `ExperimentConfig`
  field (cap, Γ-sweep, seeds, sizes, `maximize`, geometry zscore/metric, support snap/mesh/range/rounding).
- **② Design DGP** — choose K, X & S distributions, the outcome model `σ(aₖ·X+bₖ·S+cₖ)` and confounding model
  `softmax(vₖ·X+γ·dₖ·(S−E[S]))` via per-arm coefficient tables (`st.data_editor`), OR an **Advanced (Python)** tab
  with a raw `generate(n,gamma,rng)` editor. Live preview (sample draw + plots). S is the UNOBSERVED confounder.
- **③ Run** — review + execute `run_experiment` (captures stdout), saves under `results/<name>/`.
- **④ Results** — comparison plot (deployed test value vs Γ + Full-info/Best-means ceilings), per-arm usage,
  overlap-diagnostics banner, final-numbers table.

**Clean-code architecture (KEY):** `core/` is PURE LOGIC, NO streamlit — `dgp.py` (DGPSpec + structured/code →
`generate`; `build_generate`, `build_generate_from_code`, `sample_preview`, `default_spec`), `experiment.py`
(`build_experiment_config(state)→ExperimentConfig`; `run()` captures stdout; `load_results()`), `plotting.py`
(matplotlib figs). `views/` is ALL streamlit (one `render_*` per page + `components.py`). `app.py` only wires
theme+header+sidebar-nav (radio `key="nav"`)+session. core is unit-tested headless in `tests/test_application.py`.

**DGP PRESETS (2026-06-11):** `core/presets.py` ships the discrete-X study DGPs where R-OW wins —
**exp_a (3-arm), exp_b (4-arm), exp_c (4-arm)** — as **code-mode** DGPSpecs whose `generate` is ported VERBATIM
from `code/rw_implementation/experiments/discrete/dgps.py` (X on a 21-pt grid, unobserved Bernoulli S, σ(aₖ+bₖX+cₖS)
outcomes, softmax S-channel treatment). A `📦 Load a preset DGP` selectbox on the DGP page loads them (sets
mode=Advanced/code, reseeds the code editor via `st.rerun()` + popping `dgp_code`). PRESET_NOTES give the recommended
config (γ_true=3, **discretize OFF** since X is already a grid, z-score OFF, compare R-OW/R-O/IPW).
**VERIFIED exp_c reproduces the study** (new code 1.1 pipeline, n=300, 3 seeds): realised_test R-OW > R-O > IPW at
EVERY Γ≥2 (tie at Γ=1=no robustness); at the operating Γ=4.4817 (=e^{1.5}) R-OW=0.699 > R-O=0.693 > IPW=0.651,
R-OW closest to Full-info 0.75 — matches the discrete_multiarm_study finding (Wasserstein edge present in 4-arm).

**CONTINUOUS-X PRESETS added (2026-06-11):** also ported the code/ continuous studies as presets —
**conti2d** (radial 2-D 4-arm, [[conti2d_study]]), **conti_binary** (2-D binary, [[conti_binary_study]]),
**conti** (continuous-X exp_c 4-arm, [[conti_x_study]]). Their verbatim generators live as standalone
lintable files in `application/core/preset_generators/{conti2d,conti_binary,conti}_gen.py`; `presets.py`
reads them via `_load_gen(stem)` (the discrete exp_a/b/c stay inline via `_discrete_code`). Dropdown now has
6 R-OW-wins presets. **Recommended config for the CONTINUOUS ones is discretize ON** (snap continuous X to a
grid, mesh≈6) + z-score ON for the 2-D ones — the OPPOSITE of the discrete presets (discretize OFF, X already
gridded). Independently validated all 3 generate valid (X,T,Y,Ypot,mu) data; spot-checked conti2d reproduces
R-OW top (R-OW 0.7015 ≥ R-O 0.7005 ≥ IPW 0.6961 at Γ=4.4817, Uncapped; the study's wider gap is under capacity).

**Design facts worth remembering:** n_arms is sourced FROM the DGP page (`session_state.dgp.n_arms`) so config can't
disagree with the data. DGP coefficient editors are keyed by `f"out_{K}_{d}"`/`f"tr_{K}_{d}"` so changing K/d gives
a fresh table (and resets to `default_spec`). Use `width="stretch"` NOT the deprecated `use_container_width`
(streamlit≥1.50). Verified with `streamlit.testing.v1.AppTest`: all 4 pages render, nav works, DGP preview works,
and a real run (IPW+R-OW+Kallus) → Results UI renders — 0 exceptions. See [[code_1_1_overhaul]] for the engine.
