#!/usr/bin/env python3
"""Verify the C4 rationality patch on synthetic data, before trusting any campaign number.

Three invariants, all consequences of what C4 is:

  1. MONOTONE.  C4 restricts the inner MINIMISATION, so the worst case can only get better:
     objective_value(rationality=True) >= objective_value(rationality=False), always.
  2. INERT AT Gamma = 1.  The MSM box collapses to the single point {w_hat}; C4 holds there with
     equality, so it cannot bind. Both settings must return identical objectives and policies.
     (This also re-checks the standing identity IPW-O-W == IPW-O-X at Gamma = 1.)
  3. FEASIBLE ALWAYS.  w = w_hat satisfies C4 by construction, so no solve may fail.
"""
import sys, importlib.util
import numpy as np

ROOT = "/home1/haghim/code 1.1"
sys.path.insert(0, ROOT)
import common


def LD(rel, fn):
    s = importlib.util.spec_from_file_location(fn, ROOT + "/" + rel)
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


S = {
    "IPW-O-X": (LD("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped"), "ipw"),
    "DR-O-X": (LD("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py",
                  "solve_doublyrobust_o_x_uncapped"), "dr"),
    "IPW-O-W": (LD("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped"), "ipww"),
    "DR-O-W": (LD("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py",
                  "solve_doublyrobust_o_w_uncapped"), "drw"),
}


def make(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    u = rng.choice([-1.0, 1.0], n)
    e = 1.0 / (1.0 + np.exp(-(1.2 * x + 0.5 * np.log(4.0) * u)))
    T = (rng.uniform(size=n) < e).astype(int)
    Y0 = rng.choice([-1.0, 1.0], n)
    Y1 = Y0 + 1.5 * (1 / (1 + np.exp(-(3 * x))) - 0.5) + rng.normal(0, 0.1, n)
    Y = np.where(T == 1, Y1, Y0)
    Y = (Y - Y.mean()) / (Y.std() or 1.0)                 # the runners' centring protocol
    return x.reshape(-1, 1), T, Y


def solve(name, X, T, Y, G, rat, eps=None):
    fn, kind = S[name]
    w, _ = common.ipw_weights_from_data(X, T, 2)
    mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
    kw = dict(n_arms=2, Gamma=G, discretize=False, lipschitz=3.0, rationality=rat)
    if kind == "ipw":  return fn(X, T, Y, w, **kw)
    if kind == "dr":   return fn(X, T, Y, w, mu, **kw)
    kw.update(zscore=False, epsilon=eps)
    if kind == "ipww": return fn(X, T, Y, w, **kw)
    return fn(X, T, Y, w, mu, **kw)


fails = 0
print("%-9s %6s %14s %14s %12s  %s" % ("method", "Gamma", "obj rat=off", "obj rat=on", "diff", "verdict"))
print("-" * 78)
for seed in (0, 1, 2):
    X, T, Y = make(200, seed)
    w, _ = common.ipw_weights_from_data(X, T, 2)
    D = common.pairwise_distance_matrix(X)
    eps = tuple(common.tight_epsilon(D, T, w, 2, is_distance=True, c_eps=1.0))
    for G in (1.0, 4.0, 20.09):
        for name in S:
            try:
                a = solve(name, X, T, Y, G, False, eps)
                b = solve(name, X, T, Y, G, True, eps)
            except Exception as ex:
                print("%-9s %6.2f  SOLVE FAILED: %s" % (name, G, str(ex)[:60])); fails += 1; continue
            d = b.objective_value - a.objective_value
            ok = d >= -1e-7                                       # invariant 1
            note = "monotone OK" if ok else "*** VIOLATED ***"
            if abs(G - 1.0) < 1e-12:                              # invariant 2
                inert = abs(d) < 1e-7 and np.allclose(a.pi[1], b.pi[1], atol=1e-6)
                note = "inert at G=1 OK" if inert else "*** NOT INERT AT G=1 ***"
                ok = ok and inert
            if seed == 0:
                print("%-9s %6.2f %14.6f %14.6f %12.2e  %s"
                      % (name, G, a.objective_value, b.objective_value, d, note))
            if not ok: fails += 1

# invariant 2b: the standing Gamma=1 identity must survive the patch
X, T, Y = make(200, 0)
w, _ = common.ipw_weights_from_data(X, T, 2)
D = common.pairwise_distance_matrix(X)
eps = tuple(common.tight_epsilon(D, T, w, 2, is_distance=True, c_eps=1.0))
ox = solve("IPW-O-X", X, T, Y, 1.0, False, eps)
ow = solve("IPW-O-W", X, T, Y, 1.0, False, eps)
dd = float(np.abs(ox.pi[1] - ow.pi[1]).max())
print("\nGamma=1 identity  max|pi(IPW-O-X) - pi(IPW-O-W)| = %.2e  %s"
      % (dd, "OK" if dd < 1e-6 else "*** VIOLATED ***"))
fails += (dd >= 1e-6)

print("\n%s  (%d failures)" % ("ALL INVARIANTS PASS" if fails == 0 else "FAILURES PRESENT", fails))
sys.exit(1 if fails else 0)
