---
name: wasserstein-2arm-cap
description: "The 2-arm Wasserstein (R-OW vs R-O) edge is enabled by the capacity cap binding, NOT by covariate dimensionality — 1-D single feature suffices."
metadata: 
  node_type: memory
  type: project
  originSessionId: b9b1bd8f-6c5e-40f9-8b50-6e13ed6d978f
---

In the 2-arm (binary-treatment) case, whether R-OW (odds-box ∩ Wasserstein) separates from
R-O (odds-box only) is governed by **whether the capacity constraint binds**, NOT by the number
of covariate dimensions.

- **Uncapped 2-arm collapses**: R-OW ≈ R-O ≈ IPW regardless of covariate dimension (the box ∩ Wasserstein
  optimum coincides with the box optimum). This is the collapse noted in [[conti_binary_study]] / [[discrete_multiarm_study]].
- **A loose cap recovers the edge even in 1-D**: a single discrete feature X (21-pt grid in [-1,1]),
  unobserved S~Bernoulli(½), control σ(0.5), treatment σ(-1.2+4X+5S), logit P(T=1)=-X+γ(S-½), γ_true=5
  (matched Γ=12.18), **cap=(1.0,0.5)** → R-OW 0.699 > Regret-OW 0.688 > IPW 0.661 > R-O 0.650 > Regret-O 0.613.
  Wasserstein edge over R-O = +0.049. The two Wasserstein methods sit above IPW; box-only R-O actually dips
  *below* IPW (odds-box robustness alone mis-allocates the capped budget into low-X cells S inflated).

**Why:** I earlier wrote in the 2-D Experiment panel that "the 1-D version collapses to R-OW = R-O / needs
>1 covariate" — that was WRONG. Corrected in index.html (2-D panel lead + new "2-arm · 1-D · R-OW wins"
Experiment subtab, assets/exp_1d/). 1-D n=800 Wasserstein Γ-sweep ≈ 9 min (~80-160s/Γ).

**Confounder DISTRIBUTION decides robust-vs-IPW (uniform-S variation).** Same 1-D DGP but unobserved
S~Uniform{0,0.1,…,1} (11 levels) instead of Bernoulli{0,1}: the extremes S=0,1 still give ±γ/2 so the
**worst-case** matched Γ=e^{γ/2}=12.18 is unchanged, BUT only ~2/11 units sit at those extremes so the
**typical** confounding is much milder. Result @ matched Γ: **IPW 0.713 ≳ R-OW 0.702** > Regret-OW 0.684 >
R-O 0.657 > Regret-O 0.617 — robustifying to the worst-case Γ **over-hedges** and IPW edges R-OW. Yet R-OW is
best across **moderate** Γ (Γ=3:0.735, 6:0.727, 9:0.722), peaking at Γ≈3 and only crossing below IPW past
Γ≈10. Wasserstein edge still holds (R-OW ≥ R-O every Γ, +0.045 at matched). Lesson: matched Γ=e^{γ/2} is a
WORST-CASE bound; the robust optimum should be calibrated to the *typical* (not worst-case) confounding —
the more concentrated the confounder, the more robustness at the matched Γ pays. index.html "2-arm · 1-D ·
uniform S" subtab, assets/exp_1d_uniformS/.

**DoublyRobust family added (2026-06-14) — R-OW-DR is the BEST method in BOTH 1-D experiments.** Added
R-OW-DoublyRobust + R-O-DoublyRobust (capped, cap=(1.0,0.5)) WITHOUT rerunning the other 6 methods, via
`assets/exp_1d/exp_1d_add_doublyrobust.py {bern|uni}` (reproduces seed-0 data, REUSES stored NPZ, has a
reproduction GATE: reconstructs a stored method's rt from its treat_grid, asserts Δ=0 — verified). DR methods
take an ESTIMATED cross-fitted μ̂_k (per-arm logistic on the binary outcome — never oracle; 85% benefit-sign
agreement). Also added a **non-robust AIPW BASELINE** (the doubly-robust analog of IPW — IPW + estimated μ̂, Γ=1 so the box
collapses, NO confounding robustness, a flat Γ-independent line; = capped DoublyRobust/R-O-DR at Γ=1) via
`assets/exp_1d/exp_1d_add_aipw.py {bern|uni}` (one Γ=1 solve, reuses NPZ). This gives a clean ABLATION:
IPW (propensity only) → AIPW (+ outcome model) → R-O-DR/R-OW-DR (+ confounding robustness).
Results @ matched Γ=12.18 (realised test E[Y], higher better):
- **bern (STRONG confounding) — robustness pays:** R-OW-DR **0.714** > AIPW 0.702 > R-OW 0.699 > R-O-DR ≈
  Regret-OW 0.688 > IPW 0.661 > R-O 0.650 > Kallus 0.617 > Regret-O 0.613. Ablation: outcome model is the big
  jump (IPW 0.661 →(+μ̂) AIPW 0.702, **+0.041**), then confounding robustness adds a further **+0.012** (AIPW
  0.702 → R-OW-DR 0.714 ≈ ceiling 0.716). Both rungs pay; R-OW-DR best + most Γ-stable.
- **uni (MILD typical confounding) — robustness OVER-HEDGES:** **AIPW 0.740** > R-OW-DR 0.715 > IPW 0.713 >
  R-OW 0.702 > R-O-DR 0.700 > … The non-robust **AIPW WINS** (≈ ceiling 0.750); robustifying it makes R-OW-DR
  *decline* from AIPW (its Γ=1 value 0.740) down to 0.715 as Γ→worst-case — the SAME over-hedging that lets plain
  IPW (0.713) edge plain R-OW (0.702), now at the DR level. (My earlier "R-OW-DR reclaims the win from IPW" was
  WRONG — AIPW is best in uni; corrected.)
**LESSON (consistent across both tabs): the outcome model is the big win; confounding robustness pays only when
the confounding is REAL — it helped in bern, over-hedges in uni.** DR > IPW-only sibling across moderate Γ in both.
AIPW color #ff7f0e (orange), marker 'p'; placed next to IPW in legends/POLICY_DATA (9 series). Recovered the lost
generators `exp_1d.py`/`exp_1d_add_kallus.py` from transcript → persisted to `assets/exp_1d/`. HTML: both 1-D
subtabs' Finding cards + captions + interactive POLICY_DATA (8 series) updated. See [[code_1_1_overhaul]] DR methods,
[[kallus_synthetic_repro]] (same DR-breaks-degeneracy story on the continuous Kallus DGP).

**JUSTIFICATION plot suite (2026-06-15, for the advisor write-up) — `assets/exp_1d/exp_1d_*.py`, added as a
"Why these results happen" section to BOTH 1-D subtabs.** Mechanism = ONE causal chain: unobserved S drives BOTH
outcome (σ(−1.2+4X+**5S**)) AND assignment (logit P(T=1)=−X+**5**(S−½)) ⇒ high-S units (whom treatment helps) are
treated more ⇒ observed treated outcome is INFLATED ⇒ naive IPW over-treats; μ̂/R-OW escape. Figures:
1. `data_diagnostics.png` (per mode, `exp_1d_data_diagnostics.py`): 4 panels — (A) who's treated+counts; (B) the
   confounding E[S|X,T=1] vs E[S|X,T=0] (gap **+0.81 bern, +0.36 uni**); (C) observed-vs-true-vs-μ̂ outcome means
   (μ̂ error 0.181 bern vs **0.089 uni** ⇒ why AIPW dominates uni); (D) the TRAP — naive CATE says treat low-X where
   true CATE<0 (7/21 cells bern, 3/21 uni).
2. `confounding_dist.png` (`exp_1d_confounding_dist.py`): per-unit Γ_i=exp(γ|S−½|). bern **100%** at worst-case
   Γ=12.18; uni only **17%** at worst-case, median Γ_i=4.5, 30% near 1. THE reason robustness pays (bern) vs
   over-hedges (uni).
3. `ablation_bars.png` (`exp_1d_justify_plots.py`): IPW→AIPW→R-O-DR→R-OW-DR decomposition @ matched Γ. **DELTAS
   (consistent across both): +outcome-model μ̂ ALWAYS the big win (+0.041 bern / +0.027 uni); +box-O ALWAYS
   over-hedges (−0.014 / −0.041); +Wasserstein-W recovers (+0.026 / +0.016); NET robustness +0.012 bern (pays) /
   −0.025 uni (over-hedges).**
4. `robustness_vs_typical.png` (`exp_1d_justify_plots.py`): realised-test-vs-Γ with worst-case matched Γ (12.18) AND
   typical median Γ_i (bern 12.18=matched; uni 4.5≪matched) marked + over-hedging zone shaded. R-OW/R-OW-DR peak near
   TYPICAL Γ, decline beyond ⇒ set robust Γ to the typical, not worst-case.
ALSO **rescaled `realized_train_test.png`** (both modes): y-axis zoomed to the methods (~[0.61,0.77] vs old [0.60,0.87],
~2× visible) with Full-info ceiling annotated off-scale. All via `exp_1d_justify_plots.py` (reads NPZ, no method re-run).
HTML lesson: NEW h4 subheads used (no CSS, browser-default — fine); used `\lt` not raw `<` (e.g. Γ_i\lt2). See [[keep_html_in_sync]].

**5-SEED run + SD bands (2026-06-15).** Ran the full suite for seeds 0–4 × both modes via
`assets/exp_1d/exp_1d_multiseed.py {mode}{seed}` → `_multiseed/{mode}_seed*.json`; aggregated to mean±SD with
`exp_1d_multiseed_agg.py` → `_chartdata.json`; the interactive realised/objective charts now shade **±1 SD bands**
(renderChart `ysd` support, see [[interactive_charts]]). Seed-0 reproduces stored NPZ exactly (gate Δ=0).
**5-seed mean±SD @matched Γ (TEST):**
- **bern:** R-OW-DR **0.717±0.006** > R-OW 0.708±0.014 > R-O-DR 0.703±0.010 > AIPW 0.694±0.012 > Regret-OW
  0.682±0.031 > IPW 0.656±0.012 > R-O 0.637 > Kallus 0.621 > Regret-O 0.617. **The 5-seed average REORDERS the
  seed-0 story**: the robust methods (R-OW-DR, R-OW, R-O-DR) top the mean; seed-0's AIPW(0.702)>R-OW(0.699) was a
  high AIPW draw — its mean is only 0.694 (4th).
- **uni:** AIPW **0.746±0.023** > R-OW-DR 0.735±0.011 > R-OW 0.723±0.019 > R-O-DR 0.716 > IPW 0.712 > … AIPW
  highest mean but LARGEST SD; R-OW-DR a hair behind with HALF the variance.
- **KEY FINDING: R-OW-DR has the smallest SD in both (±0.006 / ±0.011) — confounding-robustness buys RELIABILITY**
  (lowest seed variance), while non-robust AIPW is the highest-variance method. The Finding-card/Why numbers stay
  seed-0 (a single draw inside the bands; flagged as such in the realised captions).
- **WORKFLOW GOTCHA:** 10 parallel Wasserstein-LP jobs over-contend CPU → agents abandon (StructuredOutput never
  called, jobs killed); only 4–5 finish. Fix: re-run the missing with LIMITED parallelism (`xargs -P 3`), not 10-wide.

**RESTRUCTURED to exp_first/exp_second template (2026-06-16) — bern subtab only.** Rebuilt the `sub-exp_1d` (bern)
panel in index.html into the Capped/Uncapped **inner-subtab** layout (`showInner('exp1d-cap'/'exp1d-uncap')`, Capped
default) with a per-seed **Seed-selector** dropdown (Seed 0–2 + Average) on the interactive policy chart, mirroring
[[secondexp_2d]]/[[firstexp_case4]]. Generated a REAL **uncapped** regime (cap=(1,1)) so both inner tabs are populated.
Re-ran at **n=500, 3 seeds** (user's choice for speed; was n=800/5-seed). New/rewritten scripts in `assets/exp_1d/`:
`exp_1d_multiseed.py` (rewritten — BOTH regimes + per-seed/per-Γ treat-grids + gurobi `constrained_oracle` ceilings,
exp_first schema `_multiseed/bern_seed*.json` → `{regimes:{uncap,cap}}`), `build_policy_chart.py` (NEW — builds
`policy_{cap,uncap}.json`, splices `POLICY_EXP1D_CAP/UNCAP` + regime-nested `REALIZED_DATA`/`OBJECTIVE_DATA` into
index.html, regenerates standalone `_exp_1d_view.html`), `perf_plots.py` (NEW — per-regime `*_{cap,uncap}.png`),
`exp_1d_multiseed_agg.py` (regime-nested). One-off `_restructure_html.py` did the panel rewrite (idempotent-guarded).
Results @matched Γ (3-seed mean, TEST): **capped** R-OW-DR 0.718 > AIPW 0.715 > R-OW 0.705 > R-O-DR 0.703 > Regret-OW
0.698 > IPW 0.670 > R-O 0.646 (R-O dips below IPW) > Kallus 0.618 > Regret-O 0.617; **uncapped** Regret-OW 0.698 > IPW
0.686 ≈ AIPW 0.685 > **R-OW 0.674 (BELOW IPW)** > R-O-DR 0.672 ≈ R-O 0.671 > R-OW-DR 0.663 — robustness does NOT win
without the cap (the honest contrast, parallels [[firstexp_case4]] uncapped). best-means ceiling 0.726 ≈ full-info 0.729
**identical across regimes** because optimal treat-frac ≈0.52 only just exceeds the 0.5 cap → the cap *just barely* binds,
which is exactly enough to activate R-OW's edge. Old n=800 seed files backed up to `_multiseed/_old_n800/`. The **uni
subtab is NOT yet restructured** (still old n=800 data; the new scripts skip old-schema modes via a `"regimes" not in`
guard, and REALIZED/OBJECTIVE keep `.uni` flat so the uni panel still renders). Mechanism "Why" appendix kept (seed-0
n=800 figures) with a muted note that headline numbers are the new 3-seed n=500 run. Verified all 8 charts render
(offsetHeight>0 on tab activation) + MathJax typeset via headless Chrome. Legacy `exp_1d.py`/`exp_1d_add_*.py`/
`exp_1d_capture_grid.py`/`exp_1d_build_chartdata.py` left on disk (mechanism figs) but out of the chart pipeline.
See [[keep_html_in_sync]], [[interactive_charts]].

**How to apply:** to make R-OW uniquely best in any 2-arm demo, add a loose capacity cap (cap≈0.5) — don't
reach for extra covariate dimensions. Without a cap, expect 2-arm R-OW = R-O. See [[wasserstein_binding]].
To add a new method to a saved experiment WITHOUT rerunning others: reproduce the seed-0 data, load the NPZ,
gate via stored-treat_grid→rt reconstruction (must be Δ=0), compute only the new method, re-save + regenerate.
