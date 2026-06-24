---
name: nonmono-experiment
description: "The non-monotone 1-D 2-arm DGP where R-OW/R-OW-DR beat Kallus, AIPW and IPW — design, why, pipeline."
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

New experiment **`assets/exp_nonmono/`** (HTML Experiment subtab "★ 2-arm · 1-D · non-monotone (our methods win)"),
built to a goal: a SIMPLE explainable 1-D 2-arm DGP where our free-π Wasserstein methods beat Kallus, the
doubly-robust AIPW baseline, and IPW.

**DGP (`dgp.py`, CURRENT = v1 "ctrlS-CS5": CS0=5,BA=7,BT=0.36,CS1=5,G=5,EK=5):**
X on 21-pt grid [-1,1]; S~Bern(½) UNOBSERVED. **Control Y(0)~Bern(σ(5(S−½)))** — S-DEPENDENT (marginal still ½), so
BOTH arms' observed means are biased by the same confounder (v1 change from the earlier FLAT Y(0)~Bern(½), which widened
the IPW/AIPW gaps per the user's "push confounding harder / Y0 can depend on X and U"). Treatment Y(1)~Bern(σ(7(0.36−X²)+5(S−½)))
→ **non-monotone: treatment HELPS a MIDDLE band |X|≲0.5, HURTS the edges**. Assignment T~Bern(σ(5(S−½)(1+5X²)))
→ high-S treated more, **concentrated at the EDGES** (the ×(1+5X²) factor). matched Γ=e^{2.5}≈12.18, cap=(1.0,0.5),
**n_train=500** (was 800), n_test=2000. GOTCHA fixed: `pa(X,S,0)=sig(CS0*(S-0.5)+0*X)` — the `+0*X` is REQUIRED to broadcast to an array (scalar S → scalar bug in column_stack).

**WHY each baseline fails & we win (the two ingredients):**
1. NON-MONOTONE optimal policy ("treat a middle band") ⇒ **Kallus's logistic policy σ(θᵀx̃) CANNOT represent it** →
   collapses to ≈never-treat (≈control). Also **AIPW's μ̂ is a monotone logistic in X** → can't bend to the bump.
2. EDGE-concentrated confounding ⇒ at the harmful edges the treated are a high-S elite whose observed outcomes look
   fine ⇒ **IPW and AIPW (non-robust residual) are fooled into treating the edges**.
3. **R-OW/R-OW-DR**: free-π (any band) + Wasserstein covariate-balance term flags the edge-imbalanced treated
   distribution + MSM box hedges ⇒ treat the right band. **The Wasserstein term is the differentiator: R-OW>R-O,
   R-OW-DR>R-O-DR.**

**RESULT — CURRENT = v1 @ n_train=500, 10 seeds (mean±SD @ matched Γ=12.18, TEST):**
R-OW **0.563±0.014** ≈ R-OW-DR 0.562±0.012 > R-O-DR 0.560±0.014 > AIPW 0.550±0.008 > R-O 0.547 ≈ IPW 0.547 > Regret-OW 0.512 >
Regret-O 0.497 > Kallus 0.495. Ceilings: control ≈0.50, **best-means 0.587**, oracle 0.617. Margins: over Kallus **+0.068**,
IPW **+0.016**, AIPW **+0.013**. Wasserstein edge R-OW−R-O = **+0.016**, ties-or-beats R-O in EVERY one of the 10 seeds (9 wins,
1 tie at seed 0; up to +0.032). At n=500 the win is clearest ON AVERAGE — R-OW tops the best non-robust baseline (AIPW/IPW) in
**8 of 10 seeds** (mean +0.013, ≈1 SD). [Prior v1 @ n_train=800, 5 seeds was: R-OW 0.576 > AIPW 0.562 > IPW 0.548 > Kallus 0.494,
Wasserstein edge +0.024 every seed — n=800 gave cleaner per-seed margins.] HTML chart data is INLINE in index.html (vars
POLICY_DATA_NM/REALIZED_DATA_NM/OBJECTIVE_DATA_NM, find by name not line#) — re-splice from `_chartdata.json` after any rerun;
`run_multiseed.py {seed}` reads `dgp.n_tr`, `agg_build.py` globs all seed*.json & uses nS dynamically. **GOTCHA: edit ONLY the
sub-exp_nonmono panel for seed/n text — exp_1d & exp_1d_uniS legitimately stay n=800/5-seed.**

**Pipeline:** `dgp.py` (importable DGP) · `run_multiseed.py {seed}` (9 methods × 6 Γ, +per-Γ grids on seed 0) →
`_multiseed/seed*.json` · `agg_build.py` → `_chartdata.json` (mean±SD bands + policy grids) · `justify_plots.py` →
data_diagnostics / policy_vs_x / ablation_bars / realized_train_test PNGs · `ground_truth_plots.py` → **outcome_means_vs_x.png
+ outcome_heatmaps_xs.png** ("Ground-truth outcome surfaces" section, exp_1d-style frame the user asked to replicate; shows the
NON-MONOTONE treatment bump beats control only for |X|<0.6, and BOTH arms S-split — i.e. explains why Kallus/AIPW/IPW fail). Search scripts: `explore.py`, `tune.py`,
`check.py`. HTML: interactive realized/objective charts (SD bands) + policy chart (Γ-selector) via the shared
renderChart ([[interactive_charts]]); DGP card, Finding card, Why section. **GOTCHA:** running 5 multi-seed jobs (each
with 18 Wasserstein LP solves) even at `xargs -P 3` is SLOW (~20 min) — limit parallelism, don't go wide.
**UNCAPPED companion tab** (`sub-exp_uncap`, button "2-arm · 1-D · non-monotone (UNCAPPED)", amber #b45309; `assets/exp_uncap/uncap_build.py`
→ propensity.png / whats_seen.png / realized_uncap.png / policy_uncap.png + uncap_results.json): SAME v1 DGP, cap=(1,1), so Kallus
is on equal footing. RESULT (8 seeds, matched Γ): AIPW 0.513 ≈ Regret-OW 0.511 > R-OW 0.502 ≈ R-O ≈ IPW 0.502 > R-OW-DR 0.496 ≈
Kallus 0.495 — ALL ≈ never-treat 0.498, NONE near best-means 0.586; value methods treat 97%; Wasserstein edge gone. WHY R-OW treats
everyone (the user's question, explained with plots): edge-concentrated selection → near positivity violation (treated≈all S=1,
control≈all S=0) → OBSERVED control is a low-S subset (~0.1) ≪ TRUE control 0.50 → observed data says "treat ≳ control EVERYWHERE";
robustness deflates both arms together + can't conjure missing S=1-control data → no evidence treating is bad → with no budget it
treats all → harmful edges cancel good middle → ≈never-treat. The CAP is the ranking device that makes R-OW target the band & win;
removing it (fair to Kallus) collapses differentiation for everyone. Decided via AskUserQuestion: redesign-for-uncapped-win is
STRUCTURALLY INFEASIBLE (value objective flat→treat-all; clean-control kills the signal; regret objective selective but weak ~0.51).
See [[wasserstein_2arm_cap]] (the monotone 1-D experiments), [[wasserstein_binding]], [[rw_gamma_invariance]].
