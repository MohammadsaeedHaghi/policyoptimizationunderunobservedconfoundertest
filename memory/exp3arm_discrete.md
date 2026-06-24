---
name: exp3arm-discrete
description: "3-arm discrete-X exp_a (γ=5, capacity) reproduced in code 1.1 with the current methods — new clean static subtab."
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

**`assets/exp_3arm/`** — HTML Experiment subtab **"★ 3-arm · discrete · capacity (exp_a)"** (`sub-exp_3arm`, teal #0d9488,
button after `exp_uncap`). A faithful port of the SIBLING codebase's discrete study **`exp_a` (3 treatments, γ=5,
capacity-constrained)** run with the **code 1.1 methods (all except Kallus)**, 5 seeds. User wanted it VERY CLEAN: explanations
in LaTeX (not prose), ALL plots STATIC (no renderChart/dropdown/Γ-selector), section order = (1) DGP + all DGP plots, then
(2) run + all performance plots; must include the treatment-assignment plot.

**DGP (`dgp.py`, ported from `code/rw_implementation/experiments/discrete/dgps.py` DEFAULTS["exp_a"] + run.py):** K=3;
X on 21-pt grid; S~Bern(½) unobserved. P(Y^k=1|X,S)=σ(a_k+b_kX+c_kS), a=[0,0,0], b=[-1,0,1], c=[0,0.9,1.8]; assignment
P(T=k|X,S)=softmax(β_kX+γ(S-½)d_k), β=[-.5,0,.5], d=[-1,0,1], **γ=5** (the user's variant; OPERATING default is 3.0).
μ_k=½(p|S=0)+½(p|S=1). **Capacity cap_from_train**: arm 0 uncapped, arm k≥1 capped at train share P̂(T=k)≈(1,.08,.46).
matched Γ=e^{2.5}≈12.18; n=500/2000. NOTE: dgps.py top docstring wrongly says exp_a="2 treatments" but DEFAULTS is
authoritatively n_arms=3. Confounding: S=0→arm 0, S=1→arm 2 (stark); arm 2's observed outcome inflated by high-S selection.

**code 1.1 methods are FULLY K=3-ready** (all capped solvers + common machinery, no 2-arm hardcoding; cap=(c0,c1,c2), arm 0 =
uncapped control; deploy Σ_k π_k·Ypot[:,k]). Constrained ceilings (Full-info, Best-means) via a small Gurobi LP in run_multiseed.py.

**RESULT (5 seeds, realised TEST):** @Γ=3 (peak): AIPW 0.672 ≈ R-OW 0.670 > Regret-OW 0.663 > R-OW-DR 0.661 > R-O-DR 0.651 >
IPW 0.650 > R-O 0.648 > Regret-O 0.593. @matched Γ=12.18: AIPW 0.672 > R-OW-DR 0.660 > R-O 0.658 > R-OW 0.657 > IPW 0.650 >
R-O-DR 0.621 > Regret 0.509 (collapse to control). Ceilings: full-info 0.700, best-means 0.697. HONEST: R-OW peaks at Γ=3
(Wasserstein edge R-OW−R-O=+0.022, beats IPW +0.020), reverts/over-conservative at matched Γ — faithfully reproduces the
code/ README finding; AIPW (the strong DR baseline) marginally best. NOT promised as an R-OW win.

**Pipeline:** `dgp.py` · `ground_truth_plots.py` → outcome_surfaces / outcome_heatmaps / **assignment_rule** / whats_seen ·
`run_multiseed.py {seed}` → `_multiseed/seed*.json` (+constrained-oracle ceilings, seed-0 policy grids: `policy_grid` matched-Γ +
`policy_grid_byG` per-Γ + `policy_grid_fixed`) · `perf_plots.py` → realized_vs_gamma / objective_vs_gamma / policy (static argmax-arm
strip, NOT in HTML) / realized_bar (Γ=3 & matched) / capacity_usage + `exp3arm_results.json`.

**The chosen-arm/policy plot is the ONE INTERACTIVE chart, now a RANDOMIZED-PROBABILITY small-multiples view** (user: "don't just
show the argmax arm — policies are randomized, draw 3 lines per arm / 3 panels side by side"). `build_policy_chart.py` →
`POLICY_PROB_3ARM` and splices `renderChartPanels('ichart-3arm-policy',POLICY_PROB_3ARM)` (replaces the old argmax `renderChart`
call + `POLICY_DATA_3ARM` var; the build script's regex handles first-run OR re-run via `POLICY_(DATA|PROB)_3ARM`). **Three
side-by-side panels** (one per arm: arm0 control·uncapped / arm1 ≈7% / arm2 ≈45% caps from seed0), each plotting π(arm k|X)∈[0,1],
**sharing ONE Show-methods toggle + ONE Γ-selector**. Data = `policy_grid_byG`/`policy_grid_fixed` rows (not argmax). **defaultGamma=3.0**
(value-peak — randomization clearest, e.g. R-OW arm-2 ramps smoothly 0→1; at matched Γ=12.18 policies sharpen to ~0/1). Default-hidden
(de-clutter, toggle on): R-OW-DR, R-O-DR, Regret-OW, Regret-O, Full-info → 5 shown (Best-means, R-OW, R-O, AIPW, IPW).

**New renderer machinery in index.html** (reusable): `renderChartPanels(id,D)` + `setPanelsGamma` + `_panelData`; D fields =
{x, arms, armTitles, armYlabel, gammas, defaultGamma, matchedGamma, legend, defaultHidden, byGammaArm:{gkey:[arm0_series,arm1_series,
arm2_series]}, note}; renders N `.pchart-wrap[data-arm=k]` panels in a `.ppanel-grid` + shared legend. **`setSeriesVis` generalized
querySelector→querySelectorAll** on `g[data-method]` so one toggle hides a series across all 3 panels (backward-compatible: single-SVG
charts have 1 match — verified no regression on ichart-bern/uni/nm-policy/etc.). CSS `.ppanel-grid/.ppanel/.ppanel-title`; chart div
widened to max-width:1180px. gkey(g)=str(int(g)) if g==int(g) else str(g). All OTHER 3-arm plots stay static.

**VIEWER-SYNC GOTCHA (user "I'm not seeing anything"):** the user opens the STANDALONE viewer copy **`_exp3arm_view.html`** (= a full
copy of index.html + a trailing `<script>` that auto-activates tab-experiment + sub-exp_3arm on load), NOT index.html directly. Editing
index.html alone leaves the viewer STALE → user sees the old chart. `build_policy_chart.py` now **auto-regenerates `_exp3arm_view.html`**
at the end (rpartition on the last `</body>`, re-inserts the auto-open INJ script). The other tabs have their own viewers
(`_synthlalonde_view.html`, `_uncap_view.html`, `_nonmono_view.html`) — any index.html edit touching those tabs must regenerate the
matching viewer too. Also tell the user to hard-refresh (Cmd+Shift+R) to clear browser cache. See [[keep_html_in_sync]].
Viewer copy `_exp3arm_view.html`; canonical index.html. See [[discrete_multiarm_study]] (the code/ source), [[nonmono_experiment]], [[rw_gamma_invariance]] (the over-conservative-at-matched-Γ pattern).

**GOTCHA — two different "best" plots (user flagged the apparent contradiction):** the `outcome_surfaces.png` strip = the
**UNCONSTRAINED** argmax μ_k (crossover X≈−0.4: arm 0 for X≤−0.5, arm 1 sliver at −0.4, arm 2 for X≥−0.3 = **67%** of units).
The **Best-means line in the interactive policy chart** = the **CAPACITY-CONSTRAINED** oracle (`constrained_oracle(X,μ,cap)`).
Since arm 2's unconstrained-optimal region (67%) **exceeds its cap (~45%)**, the constrained Best-means **rations arm 2 to the
highest-X units** (largest μ₂) and falls back to uncapped arm 0 elsewhere, so its crossover shifts to **X≈0** and deployed arm-2
usage = 0.41 ≈ the 0.45 cap (the budget binds). Both correct — different questions (intrinsic-best vs best-under-budget). Fixed
the strip label to "best arm (NO cap)" + clarified both figcaps so the distinction is explicit.
