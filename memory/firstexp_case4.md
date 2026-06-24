---
name: firstexp-case4
description: "'First experiment' = Case 4 discrete-X 2-arm job-training DGP; new index.html tab with Uncapped/Capped inner-subtabs."
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

**`assets/exp_first/`** — HTML Experiment subtab **"★ First experiment · Case 4 (discrete, 2-arm)"** (`sub-exp_first`,
violet #7c3aed, the FIRST subtab button in the Experiments tab). User-provided DGP ("Case 4 = case 1 but discrete"),
binary treatment (arm0 = no training / control, arm1 = training), 5 seeds, n_train=500.

**DGP (`dgp.py`):** X~Unif on 21-pt grid [-1,1] (employability), S~Bern(½) UNOBSERVED (perseverance). Outcomes (binary):
P(Y⁰=1|X,S)=min{σ(X)+S/2,1}; P(Y¹=1|X,S)=min{σ(X+5)+S,1} if X≥0 else min{σ(X−5)+S,1} (**the +S/2, +S boosts are OUTSIDE
the sigmoid** — additive, not σ(X+S/2); a user mis-read this as σ(0.5)=0.62, correct E[Y⁰|X=-1]=0.52). Propensity
π¹=clip(σ(X+γS−2),0.05,0.95). **NON-MONOTONE: training helps X≥0, hurts X<0 → optimal = treat iff X≥0.** μ₀ rises 0.52→0.87;
μ₁=0.50 for X<0, =1.0 for X≥0. E[Y]: never=0.719, treat-all=0.762, optimal=0.817.

**γ=2 CHOSEN by `screen.py`** to satisfy the user's two requirements: (a) R-OW beats IPW in both regimes, (b) X plays a real
role in assignment (π¹ X-spread ≈0.46 at S=1). γ≥2.5 FAILS uncapped (R-OW treats-all, loses to IPW); γ=5 makes S dominate
(X-spread 0.07). matched Γ=e^{γ/2}=e≈2.7183; sweep [1,1.5,2,2.7183,3,4.35,7.07] (dgp.gammas_for). Run BOTH regimes:
UNCAPPED cap=(1,1) + CAPPED cap=(1,0.5) (user asked for both).

**RESULT (5 seeds, realised TEST @matched Γ):** UNCAP — DR methods strongest (R-O-DR 0.776, R-OW-DR 0.774), R-OW 0.760 > IPW
0.753 (small edge; AIPW 0.763), Kallus 0.742. CAP — **R-OW BEST 0.776** > Regret-OW 0.766 > DR 0.75-0.76 > IPW 0.738 ≈ AIPW
0.737 (best-means 0.813). KEY honest caveat: at X<0 every OBSERVED treated unit is a high-S elite with Y≈1 (saturated), so the
worst-case box has no low outcomes to up-weight → uncapped robustness can't correct the edge at matched Γ (only flips as Γ grows;
R-OW value RISES with Γ). The cap is what lets R-OW win (rations budget toward X≥0). Per-seed policies noisy (seed0: R-OW tilts
X≥0, but IPW also tilts X≥0, AIPW/R-OW-DR invert) — the aggregate VALUE is the robust signal, not the per-X story.

**Pipeline:** `dgp.py` · `screen.py` (γ pick) · `ground_truth_plots.py` → potential / average_potential / assignment_rule /
inverse_propensity / whats_seen (5 DGP plots, matches user's referenced figs) · `run_multiseed.py {seed}` → `_multiseed/seed*.json`
(both regimes, 9 methods incl Kallus + constrained-oracle ceilings; seed0 saves per-Γ treat-grids) · `perf_plots.py` → per-regime
realized_vs_gamma / realized_bar / objective_vs_gamma / treat_fraction (×2 regimes = 8 plots) + `exp_first_results.json` ·
`build_policy_chart.py` → two interactive treat-prob charts `ichart-first-{uncap,cap}-policy` (standard renderChart, π(treat|X)
per method, Γ-selector) spliced + **regenerates `_exp_first_view.html`** (viewer auto-opens sub-exp_first). Run seeds with `xargs -P 3`.
**The policy chart plots the MEAN π(treat|X) over all 5 seeds** (user asked, not seed 0): run_multiseed.py saves per-X treat-grids
for EVERY seed (the `if SEED==0` guards on grids were removed → `if True`), build_policy_chart.py averages across seed*.json.
A fractional value = the fraction of seeds that treat at that X (= policy stability). KEY teaching payoff (user's X>0 question):
uncapped IPW mean(X≥0)=0.75 (noisy, dips to 0.40 at X=0.3,0.4 — per-X finite-sample flips: lucky high-Y control units +
nC>nT + IPW compares weighted SUMS), AIPW 0.96, **R-OW & R-OW-DR = 1.00 (clean — Wasserstein pools across X)**, Best-means = 0/1
threshold at X=0. Same root cause as the non-monotone IPW chaos (no pooling); single-seed flips are noise (5-seed IPW value 0.753≈treat-all).

**SEED SELECTOR (user request, no re-run — used the saved per-seed grids):** the two policy charts now have a THIRD dropdown
(Seed 0 / … / Seed 4 / **Average**, default Average) beside Show-methods + Γ. build_policy_chart.py emits
`seriesBySeedGamma:{seedkey:{gkey:[series]}}` (seedkeys "0".."4"+"avg") + `seeds:[{key,label}]` + `defaultSeed:"avg"`.
NEW reusable renderChart extension (backward-compatible — guarded by `if(data.seriesBySeedGamma)`, other charts unaffected):
`function setChartSeed(id,sk,btn)` swaps `reg.data.seriesByGamma=seriesBySeedGamma[sk]` then re-renders at the current Γ;
renderChart sets `data._seed`/`data.seriesByGamma` from defaultSeed at the top and adds a `.seed-dd-btn` dropdown after the Γ one.
Switching seed shows that seed's raw 0/1 policy; Average shows the smooth fractional mean.

**INNER-SUBTAB SPLIT (user request):** §1 DGP shared at top; then a third-level tab bar splits Performance into
**Uncapped / Capped**. NEW reusable mechanism in index.html: `function showInner(id,btn)` (scopes to `.subpanel`, toggles
`.inner-panel`/`.inner-btn` active) + CSS `.innertabs/.innertabs-label/.inner-btn/.inner-panel` (violet active pill). Structure:
`<div class="innertabs">` (2 buttons) + `<div id="first-uncap" class="inner-panel active">…</div>` + `<div id="first-cap"
class="inner-panel">…</div>`; finding card shared below both. This is the ONLY 3rd-level tab in index.html (others use showTab/showSub).

See [[exp3arm_discrete]] (sibling 3-arm tab), [[nonmono_experiment]] (the other non-monotone 2-arm), [[keep_html_in_sync]]
(regenerate the `_*_view.html` viewer on any edit — build_policy_chart.py does it for exp_first).
