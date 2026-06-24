---
name: conti_x_study
description: "conti-X study — 4-arm exp_c, continuous X, capacity; RW family (Shapley/KNN) > Direct IPW vs Oracle."
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

Third Experiments sub-tab **"conti-X study"** (`#exp-conti`, 2026-06-05): the discrete `exp_c`
(**4 treatments**) with **continuous X** (X∼Uniform[-1,1]) and its **capacity constraint**. Methods:
**RW, RW−W, Direct IPW** — free-π capacity-constrained LPs (`solve_*_multiarm`, cap=historical per-arm
shares, control uncapped) — each **extended off-support via BOTH multi-arm Shapley and KNN** — plus
**Oracle** (capacity-constrained per-unit test ceiling). **No Kallus** (parametric Kallus can't enforce
capacity; free-π does, exactly). 5 seeds, γ_true∈{3,5}.

**KEY GOTCHA — bin the policy.** With *truly* unique continuous X, the free-π policy is a per-unit
singleton → degenerate/overfit and **all methods collapse to the same value** (saw RW=RW−W=DI=0.584,
Γ-invariant). Fix: `POLICY_ROUND=1` → `rounding_digits=1` in the solvers' `tie_same_x` bins X into ~21
cells (many units/cell → the LP has covariate shift to balance), then extend to continuous test. This
**restored the separation**.

**Result (5 seeds, matched Γ=e^{γ/2}):** RW > RW−W > Direct IPW (Γ-responsive, like discrete 4-arm).
γ=3 (matched 4.48): RW-Shapley 0.664 > RW−W-Shapley 0.655 > Direct IPW 0.638 (KNN similar). γ=5
(matched 12.18, far out): RW≈RW−W≈0.66 > DI 0.65 — RW beats RW−W at its *peak* (lower Γ); RW−W catches
up at huge Γ (same as discrete). Oracle ≈0.971 (per-unit 4-arm max — unreachable; the big gap is
irreducible binary-outcome noise, not method weakness). Shapley ≈ KNN (small extension noise).

**Code:** `experiments/conti/{dgps.py (4-arm exp_c continuous, reuses discrete exp_c coeffs + a
true_best_arm),run.py}`. New **multi-arm extension** `srpo/extension/multiarm.py`:
`extend_with_{shapley,knn}_multiarm(X_test, X_train, pi (K,n))` = per-arm extend + renormalise to the
simplex (exported in `srpo`; test `test_multiarm_extension_simplex_and_support`). run.py: per
(γ,seed,Γ) fit RW/RW−W/DI (cap, rounding_digits=1) → extend (K,n) policy Shapley&KNN → realised
train/test + per-arm usage; Oracle constrained; parallel over (γ,seed), per-γ incremental save +
flushed prints. Plots: 2×2 (train,test,worst-case obj, chosen-arm-vs-x) + per-arm capacity-usage (2×K).
Artifacts `experiments/conti/out/exp_c_{gt3_00,gt5_00}.{csv,png}` + `exp_c_panel_*.npz`. 42 tests pass.

**HTML:** `build_html.build_conti()` → `#exp-conti` sub-tab (exp_c 4-arm; variant tabs
[★ Compare all, Capacity γ=3, Capacity γ=5]); guarded/idempotent splice + nav button.
**Path gotcha (fixed):** `CONT_OUT`/`CONTI_OUT` need `OUT.parent.parent/<pkg>/out` (OUT is
`.../experiments/discrete/out`).

**History:** first tried 2-treatment binary (user: "use shapley") but in 2 arms RW≡Direct IPW
structurally ([[rw_gamma_invariance]]) — can't separate; user then allowed ≥3 arms for the RW edge, so
the binary conti was replaced by this 4-arm version. The earlier all-method **binary continuous w/
parametric Kallus** (`experiments/continuous/`) stays TERMINATED/dormant (Kallus-W hung at large Γ).
See [[discrete_multiarm_study]].
