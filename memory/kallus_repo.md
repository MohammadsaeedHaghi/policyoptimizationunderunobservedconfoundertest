---
name: kallus_repo
description: "File map, dependencies, and known issues of the official CausalML/confounding-robust-policy-improvement repo (Kallus & Zhou 2021 reference code)."
metadata: 
  node_type: memory
  type: reference
  originSessionId: 85317083-7452-4dfa-ba5f-16d73019ebb5
---

**Repo:** https://github.com/CausalML/confounding-robust-policy-improvement
**Local clone:** `paper/kallus/repo/` (created by us; not user-supplied).
**Status:** Last pushed 2024-01-31, ~7 files, ~120KB total. Public, no license.

**File map.**
- `methods.py` — `ConfoundingRobustPolicy` wrapper. `.fit(X, T, Y, q0, GAMS, method_params, eval_conf)` iterates over Γ, calls the chosen optimizer, runs **reversion check** (use a previously-trained policy from a smaller Γ if it scores better on the current Γ — uses nesting of uncertainty sets) and **truncation** (revert to baseline if objective > tol). Also wraps optimal-tree policies via `get_opt_tree_policy`, but that path is broken (depends on missing files).
- `unconfoundedness_fns.py` — Helpers. Key functions: `get_bnds(q, logGamma) → [a_bnd, b_bnd]` (marginal sensitivity model); `find_opt_weights_short/_shorter` (Theorem 3 sort-based inner solver — linear scan vs ternary search); logistic-policy probabilities for binary and multinomial cases; Gurobi LPs for TV-budgeted inner subproblem.
- `subgrad.py` — Subgradient routines. `grad_descent_sharp` is the per-treatment-normalized "sharp" version that matches the paper (binary case). `grad_descent_sharp_mt` is the multi-treatment version. `Rbar`/`Rbar_mt` evaluate worst-case regret at given Γ. Armijo line search. `opt_w_restarts` runs N restarts and picks best.
- `data_scenarios.py` — Synthetic DGP. `generate_log_data` (binary) and `generate_log_data_mt` (multi-treatment) construct data where true propensity hits the marginal-sensitivity-model upper bound when treatment matches the CATE-optimal action.
- `methods_test.py` — Replication driver for the synthetic experiments. CLI: `python methods_test.py N_REPS gam1,gam2,...`.
- `WHI_eval.py` — WHI case-study driver (requires WHI data pickles not in repo).
- `README.md` — Two-paragraph usage sketch.

**Pitfalls / issues to know before treating it as ground truth.**
1. **Python 2 syntax throughout** (`print x`, no parens). Won't run on Python 3 without translation (`2to3` mostly works).
2. **Missing files referenced by imports:** `opt_tree.py`, `greedy_partitioning.py`, `greedy_partitioning_serv.py`, `scripts_running.py`. Means the decision-tree policy path is non-functional; logistic-policy path works in isolation.
3. **Hard-coded paths:** `data_scenarios.py` line 16: `module_path = '/Users/az/Box Sync/unconfoundedness'` (author's machine).
4. **Heavy deps:** `gurobipy` (commercial, needed for budgeted/TV uncertainty set; binary unbudgeted case only needs numpy/sklearn), `cvxpy`, `sklearn`, `joblib`, `matplotlib`.
5. **Duplicate definitions:** `get_implicit_grad_centered`, `centered_around_p1`, `find_opt_weights_short` defined in both `unconfoundedness_fns.py` and `subgrad.py`. Import order matters — `from X import *` then `from Y import *` lets later one win.
6. **Subtle: in `grad_descent_sharp` the subgradient divides by `np.sum(wghts_total)`** (sum of two per-treatment-normalized vectors = 2) rather than per-treatment normalization at the gradient stage. Effectively a constant scaling of step size; not a correctness bug for direction but worth noting if matching numbers.
7. **Optimization refinements** (reversion + truncation) are mentioned in paper §6 and Appendix C.1, not in Algorithm 1 as written. Reimplementation should treat them as standard, not optional.
8. WHI data files (`WHI_OS.pkl`, `WHI_CT.pkl`) are *not* in the repo — would need to be obtained separately to replicate Sec 7.2.

**Why this matters.** The repo is the only public reference implementation, but it is research code: incomplete, Py2-only, Gurobi-dependent. Treat its high-level structure as the implementation guide (per-Γ loop, sort-based inner solver, sharp gradient, restarts + Armijo, reversion + truncation), but expect to rewrite cleanly in modern Python.

**How to apply.** When the user asks for an implementation, mirror the *algorithmic* structure (Algorithm 1 + Theorem 3) and the practical refinements (reversion / truncation / restarts) but write fresh Python 3 code under `code/`; don't try to port `methods.py` line-for-line.

See also [[kallus_paper]] (formulas/notation).
