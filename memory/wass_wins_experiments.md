---
name: wass-wins-experiments
description: "exp_wass (continuous-X) + exp_wass_disc (discrete-X) \"Wasserstein wins\" experiments — ε>0 is a propensity property, not a continuum property"
metadata: 
  node_type: memory
  type: project
  originSessionId: 58aeb24f-db15-4091-9dfb-a7f743c17425
---

Two index.html Experiment sub-tabs built to show WHEN the Wasserstein covariate-balance term (O-W) beats the odds-box alone (O-X). Shared DGP: X covariate, unobserved S correlated with X via P(S=+1|X)=σ(4X), e(X,S)=σ(S−1.8X) (mis-targeted+S-confounded), μ0=S, μ1=S+X ⇒ CATE=X ⇒ oracle = treat iff X>0, NOISE=0.4. Methods always FIT a logistic P(T|X) (never true e). Matched Γ≈e²≈8.

**Fourth experiment · continuous-X** (`assets/exp_wass/`, ★ "Fourth experiment" subtab, charts `ichart-wass-*`, var WASS_CHARTS): X~Unif(-1,1). N_train=1000, 1 seed, Γ={1,2,4,8} (W skips Γ=1=parent X-X). ε=(0.0147,0.0158). DR-O-W best at Γ=2 (0.221, oracle 0.245); IPW-O-W beats IPW-O-X at matched Γ=8 (0.195 vs 0.160); box IPW-O-X peaks 0.237 at Γ=4 (under-assumes confounding). 3-seed N=500 check held peak-to-peak.

**Fifth experiment · discrete-X** (`assets/exp_wass_disc/{dgp,run_disc,build_disc}.py`, ★ "Fifth experiment" subtab, charts `ichart-disc-*`, var DISC_CHARTS, `disc_results.json`): SAME DGP, X on 9 discrete levels. N_train=1000, 1 seed, Γ={1,2,4,8}. **The point: ε>0 comes from a MIS-SPECIFIED propensity, NOT from continuous X.** Logistic ε=(0.025,0.024)>0; counting (exact per-cell Hájek) ε=0 — the [[nonmono_experiment]]/HR regime where W is moot. Result: **DoublyRobust-O-W beats DoublyRobust-O-X at every Γ>1** (DR-O-X collapses 0.262→0.132 as Γ grows; DR-O-W holds ~0.243); IPW-O-W beats IPW-O-X at matched Γ=8 (0.259 vs 0.254) and Γ=2 (0.263 vs 0.245). Oracle 0.289, IPW-X-X=DR-X-X=0.262, Γ=1 collapses all robust to X-X parent.

**KEY HONESTY CORRECTION (2026-06-19):** UNCAPPED, IPW-O-X is the OVERALL winner in BOTH (lucky Γ=4 peak: 0.237 conti, 0.283 disc, closest to oracle) because it hedges toward treat-most and Γ=4 under-assumes the true confounding. The W "win" uncapped is only CONDITIONAL: (a) DR-O-W>DR-O-X at every Γ (DR-O-X collapses to ~0.13), (b) IPW-O-W>IPW-O-X at the matched Γ≈8. Do NOT claim "Wasserstein wins" outright for the uncapped regime.

**CAPPED regime (treat≤50%, the oracle's own frac) = the genuine outright win, added 2026-06-19** (`run_wass_cap.py`/`run_disc_cap.py` + shared `build_cap.py`; both tabs restructured into Capped/Uncapped inner-panels via showInner). The cap forces a whom-to-treat choice and removes the box's hedge-to-treat-most escape:
- **Continuous capped = CLEAN OUTRIGHT WIN:** DoublyRobust-O-W=0.237 best overall (oracle 0.245); every W value beats every box value at every Γ; plain IPW-X-X craters to 0.094 (box mis-selects half). Fourth tab Capped is the headline; Capped is default-active.
- **Discrete capped = NEAR-TIE, not a clean win:** best W 0.282 (DR-O-W@Γ2) only ties best box 0.282 (IPW-O-X@Γ4 lucky peak); coarse 9-level within-level averaging softens mis-selection so IPW-X-X only drops to 0.249. Fifth tab Uncapped is default-active; Capped shown honestly as a near-tie. The clean outright win is CONTINUOUS-X-specific.

Mechanism: S tracks X, so balancing X balances hidden S → covariate-balance constraint removes bias the odds-box can't, once ε>0; the cap binding is what converts the conditional edge into an outright win (see [[wasserstein_2arm_cap]], [[firstexp_case4]]). Build scripts splice charts into index.html (renderChart, family-color/suffix-shape code). Em-dash-free, verified via headless Chrome (0 JS errors, full-doc div balance). Contrasts the HR discrete+counting tab (ε=0, box-only wins) — see [[no_true_propensities]], [[wasserstein_binding]].
