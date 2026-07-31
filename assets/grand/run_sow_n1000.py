#!/usr/bin/env python3
"""Does SHARPNESS help O-W once the group totals are actually estimable? n = 1000, gstar, L = 3.

At n = 200 adding sharpness hurt monotonically in the number of cells (0.488 -> 0.271 on gstar),
and the mechanism was visible: the constraints tightened the worst case enormously (-0.34 -> +0.87
at Gamma = 4), the solver became confident, and with ~6 units per arm per cell that confidence was
bought from noise. The open question is whether that is an n = 200 verdict or a real one -- at
n = 1000 a 10-cell partition has ~50 units per arm per cell, so the group totals the sharpness
equalities pin are genuinely estimated.

One job = one seed. Sweeps cells x Gamma x c_eps at fixed L = 3 (per the request). cells=None is
plain O-W, the control arm.

Usage: python3 assets/grand/run_sow_n1000.py --seed 3 --out assets/grand/sow1k_s3.json
"""
import sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
_W = {}
CELLS = [None, 5, 10, 15, 30]
GAMMAS = [3.0, 5.0, 8.0, 12.0]
CEPS = [0.5, 1.0]
LFIX = 3.0


def _init():
    import common
    from shapley import extract_support
    _W["common"] = common; _W["extract"] = extract_support
    s = importlib.util.spec_from_file_location("solve_ipw_o_w_uncapped",
                                               str(ROOT / "methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py"))
    m = importlib.util.module_from_spec(s); sys.modules["solve_ipw_o_w_uncapped"] = m; s.loader.exec_module(m)
    _W["solve"] = m.solve_ipw_o_w_uncapped
    sp = importlib.util.spec_from_file_location("dgp1k", str(ROOT / "assets/exp_gstar/dgp_cont.py"))
    d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d


def _shap(Xn, sX, spv, block=200):
    Xn = np.asarray(Xn, float).reshape(-1, 1)
    out = np.empty(len(Xn)); dg = np.arange(len(sX))
    for s0 in range(0, len(Xn), block):
        xb = Xn[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        S = dd[:, :, None] + dd[:, None, :]; S = np.where(S > 0, S, 1.0)
        A = (dd[:, None, :] * spv[None, :, None] + dd[:, :, None] * spv[None, None, :]) / S
        A[:, dg, dg] = spv[None, :]
        v = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0
        if ex.any(): v[ex] = spv[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = v
    return out


def run_ceps(job):
    seed, n, nte, ceps = job
    common = _W["common"]; d = _W["dgp"]; solve = _W["solve"]; xsup = _W["extract"]
    obs, _f = d.generate(n, seed); X, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(nte, seed + 1000); Xte = te["X"]; Y1, Y0 = ft["Y1"], ft["Y0"]
    w, _ = common.ipw_weights_from_data(X, T, 2)
    Dm = common.pairwise_distance_matrix(X)
    eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=ceps))
    xq = X.ravel()
    out = {}
    for nb in CELLS:
        cl = None
        if nb:
            cl = np.clip(np.digitize(xq, np.quantile(xq, np.linspace(0, 1, nb + 1))) - 1, 0, nb - 1)
        key = "owx" if nb is None else "sow%d" % nb
        out[key] = {}
        for g in GAMMAS:
            t0 = time.time()
            try:
                r = solve(X, T, Y, w, n_arms=2, Gamma=g, discretize=False, zscore=False,
                          epsilon=eps, lipschitz=LFIX, sharp_cells=cl)
                sX, spv = xsup(X, r.pi[1])
                pe = _shap(Xte, np.asarray(sX, float), np.asarray(spv, float).ravel())
                out[key]["%g" % g] = {"value": float(np.mean(pe * Y1 + (1 - pe) * Y0)),
                                      "obj": float(r.objective_value),
                                      "treat_frac": float(np.mean(pe)),
                                      "secs": round(time.time() - t0, 1)}
            except Exception as ex:
                print("FAIL %s G=%s ceps=%s seed=%d: %s" % (key, g, ceps, seed, ex), flush=True)
                out[key]["%g" % g] = {"value": float("nan")}
            print("  ceps=%.1f %-7s G=%-4g -> %s (%.0fs)" %
                  (ceps, key, g, ("%.3f" % out[key]["%g" % g]["value"]), time.time() - t0), flush=True)
    orc = d.oracle_policy(Xte.ravel())
    refs = {"oracle": float(np.mean(orc * Y1 + (1 - orc) * Y0)), "never": float(np.mean(Y0))}
    return ceps, out, refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=4000, dest="nte")
    a = ap.parse_args()
    Path("gurobi.env").write_text("Threads 1\n")
    print("Sharp-O-W n=%d seed=%d L=%g cells=%s gammas=%s ceps=%s"
          % (a.n, a.seed, LFIX, CELLS, GAMMAS, CEPS), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(CEPS), initializer=_init) as pool:
        res = pool.map(run_ceps, [(a.seed, a.n, a.nte, ce) for ce in CEPS])
    out = {"seed": a.seed, "N_train": a.n, "L": LFIX, "cells": [str(c) for c in CELLS],
           "gammas": ["%g" % g for g in GAMMAS], "ceps": CEPS,
           "cells_data": {("ce%g" % ce): o for ce, o, _r in res},
           "refs": {("ce%g" % ce): r for ce, _o, r in res}}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("saved %s (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)
    print("DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
