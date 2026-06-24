---
name: realdata-experiment
description: "Real-data experiments (LaLonde + IHDP) with hidden confounders — honest validation+sensitivity study, HTML subtab + pipeline."
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

Real-data experiment **`assets/exp_realdata/`** — TWO separate HTML Experiment subtabs (teal #0f766e): **"★ Real data · LaLonde"**
(`sub-exp_lalonde`) and **"★ Real data · IHDP"** (`sub-exp_ihdp`). Built to the user's request: "use lalonde and ACIC, treat
some features as unobserved confounders, run all methods, report on HTML" + "split into two subtabs, one per experiment, and at
the top completely explain the data + how the unobserved confounder is made." Each subtab opens with a thorough, FACT-VERIFIED
data-explanation section (what the data is, file table, ground-truth ATE, exact confounder construction, observed covariate +
binning, evaluation, config table) — produced by a draft→adversarial-verify Workflow that bash-checked every number against the
scripts + .dta/.csv files. Then a "Results" section with the findings/plots, and a per-dataset Methodology card.

**KEY HONEST FRAMING (load-bearing):** real data rarely has the *malign-direction* confounding ("a HARMFUL action dressed up
to look BENEFICIAL") that lets the robust methods actively WIN on realised value — that structure is *designed into* the
synthetic DGPs ([[nonmono-experiment]]). So this is a **validation + sensitivity** study, NOT a fresh win. Robustness is
**conservative, not corrective**: when confounding makes a GOOD action look BAD (LaLonde), the MSM box can only hedge toward
the safe never-treat option; it cannot manufacture the hidden positive effect. Reported transparently in the HTML.

**ACIC substitution:** raw ACIC-2016 download 404'd → used **IHDP** (the canonical ACIC-style semi-synthetic benchmark: real
covariates + synthetic-but-KNOWN counterfactuals mu0/mu1). Flagged in HTML; offer to rerun on real ACIC if user provides files.
Data in `assets/exp_realdata/data/`: `nsw_dw.dta` (445 experimental NSW), `psid_controls.dta` (2490), `ihdp_1..10.csv` (747×, cols treat,yf,ycf,mu0,mu1,x1..x25).

**Experiment A — IHDP.** IHDP = 25 covariates: **6 continuous (x1–x6), 19 binary (x7–x25)**. NO cap (fair to uncapped Kallus),
10 reps, Γ=2, eval realised via mu0/mu1 on 40% held-out (448/299). Raw values **scale-dominated** (rep means span 8..50) → report
SCALE-FREE **fraction of oracle gain** =(val−ctrl)/(oracle−ctrl). Confounding always WEAK (max|corr(T,x)|≈0.15) + treatment helps
~99% → optimal=treat-all → **methods statistically indistinguishable** in BOTH variants below.
- **CURRENT (live in HTML) = 4-feature** (`ihdp_run4.py` → `ihdp4_results.json`, `ihdp4_plot.py` → `ihdp4_bars.png`): per user
  "turn all continuous + 15 discrete into unobserved confounders, rerun with the 4 remaining." OBSERVE **4 binary x7,x9,x14,x21**
  (the 4 most CATE-predictive binary; ≤16 cells, deploy by 4-bit-cell nearest match), HIDE **21** (all 6 continuous incl x6 +
  15 binary). RESULT @Γ=2: R-OW-DR 93.9% ≈ AIPW 93.6% ≈ Kallus 93.4% > R-OW 90.2% ≈ R-O 90.1% > IPW 84.2% (treat-all 93.7%).
  **Γ-SWEEP** (`ihdp4_gamma.py` → `ihdp4_gamma_results.json` + `ihdp4_gamma.png`, Γ∈{1,2,3,4,6,8}; HTML "Sensitivity to Γ" section):
  at Γ=1 the box is trivial so R-OW=R-O=IPW=84%; as Γ↑ the robust value-methods CLIMB to ≈91% (hedging nudges toward near-optimal
  treat-all); R-OW-DR best & steadiest ≈94% flat; Kallus stable ≈93% then COLLAPSES to 78% at Γ=8 (parametric softmax over-hedges);
  AIPW flat 93.6 / IPW flat 84.2 (both Γ-indep). NOTE for any future Γ-sweep: Kallus realised value must be computed directly from
  predict_kallus (parametric, not the cell-deploy `fr(pi)` path) — else IndexError.
- **PRIOR (single-cov) = x6** (`ihdp_run.py` → `ihdp_results.json`, `ihdp_normalize.py`): observe x6 (top CATE driver) binned to 10,
  hide x1-5,7-25. RESULT: AIPW=R-OW-DR 93.8% > R-OW 91.7% ≈ R-O 91.5% > Kallus 90.8% > IPW 86.1%.

**Experiment B — LaLonde** (`lalonde_run.py` → `lalonde_results.json`): TRAIN=observational (185 NSW trainees + 420 PSID
controls, strongly confounded); HIDE re74,re75 (+all but education); observe EDUCATION binned (≤8,9,10,11,12,≥13). EVAL on the
randomised experimental NSW via IPW (e known→unbiased), 5 seeds. **Famous bias: naive obs ATE = −$15,207 vs experimental
+$1,794** (10× hidden-earnings gap). RESULT (realised $1000s, refs ctrl/never-treat 4.555, treat-all/truth-optimal 6.349):
**every method hedges to never-treat.** DR methods (AIPW, R-OW-DR, Kallus) refuse to treat at ALL Γ (=4.555, poisoned μ̂);
IPW-value methods (R-OW,R-O,IPW) treat a sliver at Γ=1 (5.085) then hedge to ctrl as Γ↑. NONE reach 6.349. The confounding also
**mis-targets**: at Γ=3 IPW treats edu 9,11 but SKIPS ≥13 where the true benefit is largest (+$7,150). ATE-by-edu-bin
[−0.25,+0.37,+1.00,+2.66,+1.68,+7.15]. Lesson cards: robustness conservative-not-corrective; Γ-sweep makes uncertainty explicit;
a confounded outcome model HURTS (DR refuses all, IPW-value at least captures a sliver).
- **VARIANT (`lalonde_run4.py` → `lalonde4_results.json` + `lalonde4_realized.png` + `lalonde4_confounding.png`; HTML "Variant"
  section in the LaLonde subtab):** per user, hide ONLY {re74,re75,black,hispanic}, OBSERVE {age,education,married,nodegree}
  (age/edu binned to 3 → ~24 cells), Γ-swept {1,1.5,2,3,5,8}, 5 seeds. KEY GOTCHA: married/nodegree SHARPLY separate trainees
  (19% married, 71% nodegree) from PSID (87%, 31%) → observing them WRECKS overlap (IPW weights hit the clip at 40); Wasserstein
  LP also slow (~12-26s/solve, Γ=1 slowest). RESULT: verdict unchanged — all far below treat-all 6.349. IPW best (≈5.38 flat,
  treats a few cells); R-OW/R-O start =IPW at Γ=1 then HEDGE DOWN to ≈4.6 by Γ=8 (conservative price); AIPW 4.71 flat / R-OW-DR
  ≈4.4–4.7 (poisoned μ̂ + extreme weights → dips below never-treat 4.555); Kallus = never-treat. NOTE: the LaLonde subtab keeps
  BOTH the education-only primary analysis AND this variant section.

**Plots** (`realdata_plots.py`): ihdp_bars / lalonde_confounding (10× gap) / lalonde_bias (−15.2k vs +1.79k) / lalonde_realized
(vs Γ) / lalonde_policy (policy vs true gradient). HTML panel = pure images+cards (no renderChart). **GOTCHA:** MathJax here is
`\(\)`-only (no `$`) → a literal `\$` in PLAIN text renders a visible backslash; use bare `$` outside math, `\$` only inside `\(\)`.
See [[keep_html_in_sync]], [[html_raw_lessthan_bug]], [[save_all_artifacts]], [[no_true_propensities]].
