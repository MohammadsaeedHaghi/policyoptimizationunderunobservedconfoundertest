#!/usr/bin/env python3
"""Validate Sharp-O-W. Four invariants, none of them assumed -- all checked numerically.

  1. BACKWARD COMPATIBILITY   sharp_cells=None must reproduce plain O-W bit-for-bit.
  2. MONOTONICITY             a tighter uncertainty set can only RAISE the worst case:
                              obj(Sharp-O-W) >= obj(O-W) at identical Gamma, epsilon, L.
  3. GAMMA = 1 COLLAPSE       at Gamma = 1 the box is the single point W = w_hat, so the
                              sharpness equalities hold automatically and Sharp-O-W == O-W.
  4. DUAL == PRIMAL           THE decisive test. For the policy the solver returns, solve the
                              inner minimisation directly as a primal LP over (W, transport plan)
                              with box + Wasserstein + sharpness, and check it equals the dual
                              objective the solver reported. If the dual derivation (the sign of
                              delta, its objective coefficient) were wrong, this is what catches it.

Run: python3 assets/grand/validate_sharp_ow.py
"""
import sys, importlib.util
from pathlib import Path
import numpy as np
import gurobipy as gp
from gurobipy import GRB

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import common
from common.sensitivity import marginal_sensitivity_box
from common.geometry import distance_matrix
from common.gurobi_env import configure_gurobi_license


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


solve = _load("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")


def primal_inner(X, T, Y, w_hat, pi1, Gamma, eps, cells=None):
    """Brute force: min over (W, transport plans) of (1/n) sum_i W_i Y_i pi_{T_i}(X_i),
    subject to the MSM box, the per-arm Wasserstein balls, and (optionally) the sharpness
    equalities. Returns the optimal value -- the quantity the dual solver claims to compute."""
    n = len(T); K = 2
    a, b = marginal_sensitivity_box(w_hat, Gamma)
    D, _, _ = distance_matrix(X, zscore=False)
    pi = np.vstack([1.0 - pi1, pi1])
    configure_gurobi_license()
    m = gp.Model("primal_inner"); m.Params.OutputFlag = 0
    W = m.addVars(n, lb=[float(v) for v in a], ub=[float(v) for v in b], name="W")
    I = {k: np.where(T == k)[0].tolist() for k in range(K)}
    zeta = {(k, i, j): m.addVar(lb=0.0) for k in range(K) for i in range(n) for j in I[k]}
    for k in range(K):
        for j in I[k]:                                    # column marginal: arrives = W_j / n
            m.addConstr(gp.quicksum(zeta[k, i, j] for i in range(n)) == W[j] / n)
        for i in range(n):                                # row marginal: supplies 1 / n
            m.addConstr(gp.quicksum(zeta[k, i, j] for j in I[k]) == 1.0 / n)
        m.addConstr(gp.quicksum(float(D[i, j]) * zeta[k, i, j]
                                for i in range(n) for j in I[k]) <= float(eps[k]))
    if cells is not None:                                 # sharpness: group totals preserved
        c = np.asarray(cells).astype(int).ravel()
        for k in range(K):
            for jj in np.unique(c):
                idx = np.where((c == jj) & (T == k))[0]
                if len(idx):
                    m.addConstr(gp.quicksum(W[int(i)] for i in idx) == float(np.sum(w_hat[idx])))
    m.setObjective(gp.quicksum((1.0 / n) * W[i] * float(Y[i]) * float(pi[int(T[i]), i])
                               for i in range(n)), GRB.MINIMIZE)
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        return None
    return float(m.ObjVal)


def main():
    sp = importlib.util.spec_from_file_location("d", ROOT / "assets/exp_gstar/dgp_cont.py")
    d = importlib.util.module_from_spec(sp); sys.modules["d"] = d; sp.loader.exec_module(d)
    N = 60                                                 # small: the primal LP is O(n^2) vars
    obs, _f = d.generate(N, 0)
    X, T, Y = obs["X"], obs["T"], obs["Y"]
    w, _ = common.ipw_weights_from_data(X, T, 2)
    Dm = common.pairwise_distance_matrix(X)
    eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=1.0))
    cells = np.clip(np.digitize(X.ravel(), np.quantile(X.ravel(), np.linspace(0, 1, 6))) - 1, 0, 4)
    print("n=%d, eps=%s, %d cells (sizes %s)"
          % (N, tuple(round(e, 4) for e in eps), len(np.unique(cells)),
             np.bincount(cells).tolist()))
    ok = True

    print("\n[1] backward compatibility: sharp_cells=None == plain O-W")
    r0 = solve(X, T, Y, w, n_arms=2, Gamma=4.0, discretize=False, zscore=False, epsilon=eps, lipschitz=3.0)
    r0b = solve(X, T, Y, w, n_arms=2, Gamma=4.0, discretize=False, zscore=False, epsilon=eps,
                lipschitz=3.0, sharp_cells=None)
    same = abs(r0.objective_value - r0b.objective_value) < 1e-9 and np.allclose(r0.pi, r0b.pi)
    print("    obj %.9f vs %.9f  -> %s" % (r0.objective_value, r0b.objective_value, "OK" if same else "FAIL"))
    ok &= same

    print("\n[2] monotonicity: tighter set => worst case cannot fall")
    for G in (2.0, 4.0, 8.0):
        rp = solve(X, T, Y, w, n_arms=2, Gamma=G, discretize=False, zscore=False, epsilon=eps, lipschitz=3.0)
        rs = solve(X, T, Y, w, n_arms=2, Gamma=G, discretize=False, zscore=False, epsilon=eps,
                   lipschitz=3.0, sharp_cells=cells)
        good = rs.objective_value >= rp.objective_value - 1e-7
        print("    Gamma=%-4g  O-W %+.6f   Sharp-O-W %+.6f   (%+.6f)  -> %s"
              % (G, rp.objective_value, rs.objective_value,
                 rs.objective_value - rp.objective_value, "OK" if good else "FAIL"))
        ok &= good

    print("\n[3] Gamma=1: box is a point, sharpness is vacuous => identical")
    r1 = solve(X, T, Y, w, n_arms=2, Gamma=1.0, discretize=False, zscore=False, epsilon=eps, lipschitz=3.0)
    r1s = solve(X, T, Y, w, n_arms=2, Gamma=1.0, discretize=False, zscore=False, epsilon=eps,
                lipschitz=3.0, sharp_cells=cells)
    good = abs(r1.objective_value - r1s.objective_value) < 1e-6
    print("    O-W %.9f  Sharp-O-W %.9f  -> %s" % (r1.objective_value, r1s.objective_value,
                                                   "OK" if good else "FAIL"))
    ok &= good

    print("\n[4] DECISIVE: dual objective == brute-force primal inner minimum at the same policy")
    for G in (2.0, 4.0):
        for tag, cl in (("O-W      ", None), ("Sharp-O-W", cells)):
            r = solve(X, T, Y, w, n_arms=2, Gamma=G, discretize=False, zscore=False, epsilon=eps,
                      lipschitz=3.0, sharp_cells=cl)
            pv = primal_inner(X, T, Y, w, r.pi[1], G, eps, cells=cl)
            if pv is None:
                print("    Gamma=%-4g %s primal INFEASIBLE -> FAIL" % (G, tag)); ok = False; continue
            good = abs(pv - r.objective_value) < 1e-5
            print("    Gamma=%-4g %s dual %+.7f   primal %+.7f   diff %.2e  -> %s"
                  % (G, tag, r.objective_value, pv, abs(pv - r.objective_value),
                     "OK" if good else "FAIL"))
            ok &= good

    print("\n%s" % ("ALL CHECKS PASSED" if ok else "*** SOME CHECKS FAILED ***"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
