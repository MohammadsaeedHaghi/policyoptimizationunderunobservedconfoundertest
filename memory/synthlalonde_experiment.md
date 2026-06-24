---
name: synthlalonde-experiment
description: Semi-synthetic LaLonde experiment (real ages + simulated outcome) — the real-covariate setting where R-OW WINS.
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

**`assets/exp_synthlalonde/`** — HTML Experiment subtab **"★ Semi-synthetic LaLonde (R-OW wins)"** (`sub-synthlalonde`,
R-OW red #d62728), built because the user wanted to SEE R-OW be the best on LaLonde-flavoured data. On REAL LaLonde outcomes
R-OW CANNOT win (benign-direction confounding → robustness is a cost). So this is a SEMI-SYNTHETIC build (IHDP/ACIC recipe):
REAL LaLonde **ages** (pooled NSW+PSID, quantile-binned to a 21-pt grid x∈[-1,1]) + a **SIMULATED** Bernoulli "success" outcome
engineered to contain the malign-direction structure R-OW targets. **Transparently disclosed as semi-synthetic in the tab.**

**DGP (`dgp.py`, = the proven [[nonmono-experiment]] model re-skinned onto real ages):** hidden S~Bern(½) "motivation";
Y(0)~Bern(σ(5(S-½))); Y(1)~Bern(σ(7(0.36-x²)+5(S-½))) → NON-MONOTONE: training helps prime-age |x|≲0.6, HURTS young/old
edges ("lost wages"); T~Bern(σ(5(S-½)(1+5x²))) → motivated over-treated, CONCENTRATED AT THE EDGES. Observe age only, hide S.
matched Γ=e^2.5≈12.18, cap=(1,0.5), n=800/2000. NOTE: a CONTINUOUS level-shift confounder did NOT fool AIPW (it sat at
best-means) — the **Bernoulli sigmoid** structure is what makes AIPW fail; that's why we use Bernoulli.

**RESULT (3 seeds, matched Γ): R-OW 0.576 > R-OW-DR 0.562 > IPW 0.560 ≈ R-O 0.560 > AIPW 0.558 > Kallus 0.495.** R-OW near
best-means 0.589 (ctrl 0.495, oracle 0.624). Margins: IPW +0.016, AIPW +0.018, Kallus +0.081, Wasserstein edge R-OW−R-O +0.016.

**Pipeline:** `dgp.py` + `run_fast.py` (LIVE: single matched-Γ, 3 seeds, BAR chart — fast) → synthlalonde_results.json +
realized.png (bars) + policy.png + construction.png (4-panel: real age dist / true non-monotone CATE / hidden selection /
naive-vs-true CATE inversion). `run_and_plot.py` = the FULLER Γ-sweep version (4 Γ × 3 seeds) but SLOW (~12 min/seed: 21-cell
Wasserstein LP × Γ) — not yet run to completion; switched to run_fast for speed. Screens: `screen.py` (continuous, AIPW resists),
`screen2.py` (Bernoulli, wins). GOTCHA: `_synthlalonde_view.html` (project root) = a temp auto-jump viewer copy opened in browser;
canonical file is index.html. See [[realdata_experiment]] (the real-outcome LaLonde where R-OW does NOT win — this complements it).
