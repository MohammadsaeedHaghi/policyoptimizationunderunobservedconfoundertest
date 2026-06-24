---
name: rw_parametric_discrete
description: RW-parametric study — softmax parametric RW (box+W and odds) on discrete exp_a/b/c vs Direct IPW & Oracle.
metadata: 
  node_type: memory
  type: project
  originSessionId: 76bc310c-7fa3-422f-a574-7fd86cc1ceaa
---

Experiments sub-tab **"RW-parametric"** (`#exp-param`, 2026-06-06): the RW family with a **parametric
softmax** policy class instead of the free per-grid policy. Run on the three discrete experiments
exp_a/b/c (excluding N=1000), **unconstrained γ_true=3, 5 seeds, Γ≤7**. Two methods, both worst-case
VALUE maximisation by envelope subgradient over a **rich softmax policy**
`π_θ(k|x)=softmax_k(θ_kᵀφ(x))` with a **degree-6 polynomial** feature map φ(x)=[1,x,…,x⁶] (user asked
for a richer class; `srpo.parametric_multiarm.design_matrix` supports affine/poly/rbf via a picklable
`basis` spec, runner `BASIS=("poly",6)`):
- **RW-param (box+W)** — odds box ∩ per-arm Wasserstein.
- **RW-param (odds)** — odds box ∩ per-arm calibration Σ_{I_k}w=n (no transport).
Compared ONLY with **Direct IPW** (free-π) and **Oracle** (recomputed on the same draws). No Kallus,
no capacity (parametric can't enforce caps; unconstrained scope chosen by user).

**Result (poly-6, matched Γ=4.48):** RW-param(box+W) > RW-param(odds) in all three (Wasserstein edge
persists in the parametric class): exp_a 0.634>0.625, exp_b 0.650>0.637, exp_c 0.636>0.625. BUT both sit
BELOW the free-π **Direct IPW** (0.661/0.680/0.665) and the **Oracle** (0.72/0.75/0.74). At Γ=1 box+W=odds
(nesting); RW-param is Γ-responsive.

**KEY finding — richer basis barely helps.** A basis sweep (exp_c, Γ=4.48: affine 0.639, poly3 0.642,
poly6 0.642, rbf9 0.639, rbf15 0.643) vs free-π RW 0.693 / Oracle 0.738 shows the parametric gap to free-π
is dominated by the **smooth softmax + subgradient optimisation, NOT the basis degree** — richer φ moves it
only ~+0.003. poly-6 (used) is marginally better than affine (~+0.003–0.005). Don't expect a rich basis to
close the parametric→free-π gap on the discrete grid.

**Code:** new `srpo/parametric_multiarm.py`: `predict_rw_parametric_multiarm` (softmax → (n,K));
`_inner_worst_case_w(..., wasserstein)` (box+Wasserstein = binary `parametric_lp._solve_inner_worst_case_w`
transport LP generalised to range(K); box-only = box ∩ per-arm Σw=n LP — both Gurobi, inner ~0.9s at
N=500 4-arm even at Γ=7, NOT the slow blowup the parametric Kallus-W had);
`fit_rw_parametric_multiarm(..., wasserstein, n_iters=15, n_restarts=3)` with softmax gradient
`∂J/∂θ_j=(1/n)Σ_i Y_i w*_i(1[T_i=j]−π_{j,i})x̃_i`. Runner `experiments/discrete/param_run.py` (parallel
over (exp,seed), per-exp incremental save). On discrete grid the softmax is evaluated directly (NO
extension — test grid = train grid). Direct IPW realised-test via `policy_lookup` (exact). Artifacts
`experiments/discrete/out/<exp>_param/<exp>_{gt3_00.csv, results_gt3_00.png, panel_*.npz}`.
Tests: `test_parametric_multiarm_predict_and_fit`, `test_parametric_inner_box_w_slack_equals_box_only`
(44 tests pass). HTML: `build_html.build_param()` → `#exp-param` sub-tab (3 exp subsubtabs, 2×2 fig +
table); guarded/idempotent. See [[discrete_multiarm_study]], [[conti_x_study]].
