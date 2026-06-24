---
name: kallus-synthetic-repro
description: "Reproduction of Kallus & Zhou (2021) §7.1.1 binary synthetic experiment (Fig.1) — DGP, gotchas, where it lives."
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

Faithful reproduction of Kallus & Zhou (2021) **§7.1.1 "Binary Treatments"** (Fig. 1: out-of-sample policy
regret vs log Γ). Lives in `code 1.1/assets/kallus_repro/` (`kallus_repro.py`, `figure1_regret_vs_loggamma.png`,
`kallus_repro_data.npz`); shown in index.html Experiment subtab "★ Kallus §7.1.1 reproduction".

**Authoritative DGP source:** `paper/kallus/repo/data_scenarios.py::generate_log_data` (+ `unconfoundedness_fns.py`
`get_bnds`, `find_opt_weights_shorter`). Python-2; ported the math, not the code.

**DGP (exact):** U~Bern(½) unobserved; X|U ~ N((2U−1)μ_x, I_5); x̃=[X,1];
Y(t)=2.5t + β·x̃ + t·β_treat·x̃ + α·U·(2t−1) + ω·U + ε, ε~N(0,4).
μ_x=[−1,.5,−1,0,−1], β=[0,.5,−.5,0,0,0], β_treat=[−1.5,1,−1.5,1,.5,0], β_TC=[0,.75,−.5,0,−1,0], α=−2, ω=1.5.
Nominal ẽ(x)=σ(β_TC·x̃); U*=1[Y(1)<Y(0)] (lower risk=better); true e(x,U)=odds-tilt of ẽ by e^{±logΓ*} toward U*,
logΓ*=1.5; T~Bern(e(x,U)); Y=Y(T). Y is a RISK/LOSS (lower better); regret<0 = improvement over never-treat.

**Critical gotchas (cost real debugging):**
- **Paper typo:** paper writes X~N((2T−1)μ_x,I) — circular. Repo uses (2**U**−1)μ_x (U drawn first). Use U.
- **IPW-harm needs the un-tilted nominal.** IPW must use ẽ(x)=σ(β_TC·x) (the unconfoundedness-assumed model, repo
  line 130). If you instead ESTIMATE the marginal P̂(T|x) (logistic on (x,T)), IPW does FINE (−0.5) and the paper's
  harm vanishes. With the nominal, IPW over-treats → regret +0.26 (harmed) ✓.
- **Uncertainty set = odds-tilt (get_bnds), NOT our marginal-box.** Weight bounds [1/p_hi,1/p_lo],
  p_hi/lo = e^{±logΓ}·Q/(1−Q+e^{±logΓ}·Q). Their "log(Γ)" axis is this log-odds. ≠ our 1-D Γ=e^{γ/2} box.
- **Inner worst-case is CLOSED-FORM** self-normalised sort: max_{W∈[a,b]} (ΣWc)/(ΣW) → sort by c, bottom-k get a,
  top get b (rnd_k_val). No LP → 50 reps in ~50s.
- CRLogit clamps to never-treat (regret 0) when worst-case regret >0 → the rise-back-to-0 at large Γ.

**Result (n=200, 50 reps):** IPW +0.26 (harmed) · CRLogit dips to −0.20 at logΓ≈1.3 (≈true 1.5) then →0 ·
Oracle −0.88. Reproduces the IPW-harm + CRLogit-dip message faithfully.
**GRF caveat:** our direct-CATE T-learner RF finds improvement (−0.49); the paper's honest causal forest is harmed.
Flagged in the figure/caption. Not reproduced: §7.1.2 multi-treatment (Fig.2), CRLogit-L1 budgeted, §7.2 WHI.

**Our methods on this DGP** (index.html "★ Kallus DGP · our methods" subtab; `assets/kallus_methods/`,
generator `kallus_methods.py` — authoritative, regenerates all 5 PNGs + NPZ + CSV + `kalcate_data.json` +
`summary.json`). Ran R-OW/R-O/IPW/Regret-O/Regret-OW, **R-OW-DR/R-O-DR** (DoublyRobust), and `fit_kallus` on the
same DGP, 1-D-experiment plot structure (realised train|test regret, **realised-outcome levels**, objective,
deployed π(treat) vs true CATE, **per-arm π(T=a|x)**). **KEY finding — THREE tiers at matched Γ≈4.48:**
(1) the five free-π IPW methods are **EXACTLY identical and Γ-invariant** (test regret **+0.13 harmed**; treat
~40%, ~no benefit discrimination 0.42 vs 0.39) — uncapped free-π on unique continuous points collapses to one
per-point rule ("keep observed arm if its weighted outcome was good"), identical across all value/regret/Wasserstein
variants (box & Wasserstein don't change the per-point sign), NN-deployed it's uninformative.
(2) **The two DoublyRobust methods BREAK the degeneracy** — R-OW-DR = R-O-DR **exactly** (box/W still don't bite on
singletons), but their *estimated* μ̂ direct term (`common.outcome_means` per-arm LinearRegression, `cross_fit=True`,
reward units; 85% benefit-sign-agreement) is a **smooth function of X** → discriminates (treat **0.63 vs 0.37**),
test regret **−0.20**, recovering ~23% of the control→oracle gap. **DEGRADES mildly with Γ** (−0.25→−0.18, te) —
opposite to Kallus — because the discrimination is the Γ-independent μ̂ term and larger Γ only adds conservative
residual hedging. (Confirms R-OW-DR/R-O-DR work end-to-end; μ̂ ESTIMATED, never oracle — leakage-audited clean.)
(3) **Parametric Kallus still best** (treat 0.59 vs 0.14, test **−0.53**, ~60% of gap, trends down with Γ
non-monotonically −0.35→−0.62 toward oracle −0.87).
Lesson: pure-IPW free-π methods need *coupling* (capacity, or discrete cells w/ many units) to be non-degenerate on
continuous high-D X; the **doubly-robust direct term sidesteps this** by injecting a cross-unit μ̂ signal that
survives NN deployment; a fully discriminating policy still needs a *parametric* class (why K&Z use a logistic policy).

**CRLogit (repo) vs our Kallus (`fit_kallus`):** SAME algorithm (K&Z Algorithm 1, parametric softmax regret-min),
NOT bit-identical: (i) uncertainty set odds-tilt (logΓ) vs Tan MSM box (Γ); (ii) inner self-normalised sort vs
Gurobi LP w/ Hájek calibration; (iii) nominal σ(β_TC·x) vs estimated P̂(T|x); (iv) CRLogit clamps to never-treat π₀
when R>0, our softmax CANNOT represent π₀ (so no snap-back to 0 at large Γ — a caveat, not a bug). Curves match in
shape, not numbers. **EMPIRICALLY VALIDATED** (`assets/kallus_repro/kallus_ours_vs_paper.py` → `..._vs_paper.png`,
shown in the repro subtab "Does our fit_kallus reproduce the paper?"): on the SAME §7.1.1 DGP+reps, our `fit_kallus`
with NOMINAL weights TRACKS CRLogit's confounding-robust dip (both ≈−0.2 near true logΓ=1.5, corr≈+0.69, vs IPW +0.23,
oracle −0.88) ⇒ Algorithm-1 impl is faithful; with ESTIMATED weights (our default) it sits lower (≈−0.46) since
estimated P̂ removes the paper's engineered IPW-harm. x-axis cross-walk: our box Γ=e^{logΓ}.
**PROVENANCE: the experiment's Kallus IS the canonical `methods/Kallus/kallus.py::fit_kallus` — same file the runner
(`run_experiment.py:234`) uses; no fork. `kallus_methods.py` only passes a higher optimiser budget (n_iters=20,
n_restarts=4) than the runner default (12, 2); identical algorithm.**
Also generalised the index.html `renderPolicyChart` JS to an arbitrary x-range/label (data.xmin/xmax/xlabel,
default −1/1/"X") so the interactive treat-vs-CATE chart works; existing 1-D charts unaffected.
See [[kallus_paper]], [[kallus_repo]], [[weight_uncertainty_diagnostic]], [[wasserstein_2arm_cap]].
