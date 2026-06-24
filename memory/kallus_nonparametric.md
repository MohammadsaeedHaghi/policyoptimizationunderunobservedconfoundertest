---
name: kallus_nonparametric
description: Non-parametric (free-π) Kallus deployable on continuous+capacity data — KallusNonparametricPolicy wrapper.
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

`srpo.KallusNonparametricPolicy` (2026-06-07): a reusable **fit/predict** wrapper packaging the
**non-parametric (free-π) Kallus** regret-minimiser for **capacity-constrained continuous** data (built
on request for that future study). NO new solver math — it composes existing pieces:
- `fit(X,T,Y,ips_weights, *, n_arms, Gamma, epsilon=None, cap=None, D=None)`: solves
  `solve_kallus_w_multiarm` (wasserstein=True, needs epsilon) or `solve_kallus_multiarm`
  (wasserstein=False) with `cap` (per-arm capacity, enforced EXACTLY — the key advantage over the
  parametric Kallus, which can't) and `rounding_digits=bin_round` (X-binning; **bin_round=1 ⇒ ~21 cells**
  for continuous X, else the free-π policy degenerates to per-unit singletons — same lesson as conti-X).
  Stores `support_X_`, `support_pi_` (K,n), `result_`.
- `predict_prob(X_new)` → (K,n_new) simplex via `extend_with_shapley_multiarm` (default) or
  `extend_with_knn_multiarm` (off-support deployment). Init: `wasserstein`, `extension`, `knn_k`,
  `bin_round`, `shapley_method`. `objective_value` = worst-case regret (≤0, do-no-harm).

**Why this (not parametric Kallus):** for capacity+continuous, the parametric softmax Kallus can't
enforce per-arm caps (nonconvex aggregate); the free-π LP does, exactly — so the non-parametric one is
the right tool, with Shapley/KNN extension for off-support deployment. Both uncertainty sets supported.

**Verified** (`tests/test_kallus_nonparametric.py`, self-contained continuous 4-arm S-channel toy):
caps respected on train (both sets × both extensions), do-no-harm regret≤0, predict_prob is a valid
simplex, ≤ constrained-Oracle ceiling, and Γ=1 ≡ Direct IPW value (binned alike). 46 tests pass. Smoke
(γ=3): caps≈(1,.19,.17,.35), usage [.805,.038,.118,.04] (all ≤ cap), realised 0.533 ∈ (control 0.489,
Oracle 0.964).

**Run on the continuous+capacity dataset (2026-06-07):** `experiments/conti/kallus_run.py` runs the
non-param Kallus (odds & W) on conti exp_c (4-arm continuous, capacity), extended via BOTH Shapley & KNN,
vs Direct IPW + Oracle, 5 seeds, γ=3 & 5 (~18 min). Added as a 4th variant tab **"Kallus (non-param)"**
in the conti-X study (`build_html._conti_kallus_block`; `out/kallus_{gt3,gt5}_00.csv` + `kallus_cmp_*.png`).
Result (matched Γ): Kallus-W > Kallus-odds (Wasserstein edge), both BELOW Direct IPW (Kallus is the
conservative regret-minimiser — reverts toward control as Γ grows via do-no-harm), Shapley≈KNN, all
below Oracle (0.97). γ=3 matched: Kallus-W 0.595, Kallus-odds 0.55, Direct IPW 0.638; γ=5 matched (Γ=12.18,
far out) both Kallus→0.513. 46 tests pass. See [[conti_x_study]], [[discrete_multiarm_study]].
