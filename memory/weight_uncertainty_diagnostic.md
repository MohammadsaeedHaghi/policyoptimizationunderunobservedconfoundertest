---
name: weight-uncertainty-diagnostic
description: How to visualize O-vs-OW IPW-weight uncertainty per support point — static range is degenerate; use worst-case adversarial distortion.
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

For the "IPW-weight uncertainty over the support, O (box) vs OW (box∩Wasserstein)" diagnostic
(index.html 1-D panels; `/tmp/exp_1d_uncertainty.py`; assets/exp_1d{,_uniformS}/weight_uncertainty.{png,npz,csv}):

**The static per-unit weight range is DEGENERATE** — `min/max w_i` over the set gives `box = O = OW` for every
point, because any single coordinate reaches its box bound while other units absorb the calibration/transport
slack. Verified at n=200–800. So the naive "range the set allows" shows no O-vs-OW difference. **Don't use it.**

**The per-unit worst-case BAND (value-min↔max) is NON-MONOTONE** — OW can be wider than O per cell (the
balance-constrained adversary redistributes weights spatially). ~23/36 groups violated OW≤O. Don't use the band.

**What works (monotone, meaningful):** the **worst-case weight DISTORTION** `|ŵ − w*|` where `w*` is the
**robust lower-bound** (value-MINimising) adversary for that outcome class's rate, single direction. Each class
pinned by its own functional (good = min Σ_{Y=1} w, bad = min Σ_{Y=0} w), subject to box + Hájek calibration
(Σ_{I_k}w=n) for O, plus the per-arm Wasserstein transport (ε from `common.tight_epsilon`, c_eps=1) for OW.
`OW_distort ≤ O_distort` holds at ~all cells (0–1/78 violations). Result is sharply **localised**: control arm
unaffected (flat outcome, no imbalance); **treatment arm's high-X good-outcome distortion collapses** under OW
(Bernoulli ≈1.1→0.35, uniform ≈1.9→0.5) — exactly where unobserved S confounds σ(−1.2+4X+5S). This is the
mechanism behind R-OW's edge, made local.

**How to apply:** build the adversary as a small gurobipy LP (w MVar + transport z MVar; `marginal_sensitivity_box`,
`pairwise_distance_matrix`, `tight_epsilon`). w=ŵ is always feasible (attains tight ε). The Wasserstein constrains
the JOINT weight vector, never an individual weight — so any per-point "uncertainty" must be adversary/value-based,
not a static set projection. See [[wasserstein_2arm_cap]].
