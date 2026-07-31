#!/usr/bin/env python3
"""The LINEAR (halfspace) policy class on all three benchmarks, against free-pi and binned-15.

Why this matters. On the 8-D exp_hidim DGP the linear class took IPW-O-W from 0.056 (free-pi,
raw covariate) to 0.786, versus sharp's 0.414 -- the first configuration that beats sharp on
structure rather than tuning, because d+1 parameters do not care about dimension while a
partition needs b^d cells.

CAVEAT THIS RUN IS DESIGNED TO TEST. All three benchmarks here are 1-D, where "linear policy"
degenerates to a single THRESHOLD, 1{beta*x + b >= 0}. That class is:
  * well specified for gstar (marginal CATE is monotone in x -> oracle IS a threshold)
  * well specified for KZ18 on the propensity index
  * BADLY misspecified for KMZ, whose CATE is 2X + 2 - 4 sin(2X): the oracle treats a
    NON-INTERVAL set, which no single threshold can represent.
So KMZ is the honest stress test of whether the linear class generalises or only wins where it
happens to match the truth. A large loss there is the expected, informative outcome.

Arms: free-pi (L=3), binned15 (sharp's quantile cells), linear (MILP, 60 s cap, incumbent taken).
One task = one (campaign, seed). Uncapped only -- the capped solvers have no linear_policy option.
"""
import sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))

BINS = 15
CAMPS = [("gs", "assets/exp_gstar/dgp_cont.py", [1, 2, 3, 5, 8, 12, 16, 24, 32]),
         ("km", "assets/exp_msmbench/dgp_g15.py", [1, 1.6487, 2.7183, 4.4817, 7.3891, 12, 16, 24, 32]),
         ("kz", "assets/exp_kz18/dgp.py", [1, 1.6487, 2.7183, 4.4817, 7.3891, 12, 16, 24, 32])]
METHODS = ["IPW-O-X", "DoublyRobust-O-X", "IPW-O-W", "DoublyRobust-O-W"]
ARMS = ["free-pi_L3", "binned15", "linear"]
_W = {}


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def _init(dgp_path, gammas):
    import common
    _W["common"] = common; _W["gammas"] = list(gammas)
    from shapley import extract_support
    _W["extract"] = extract_support
    sp = importlib.util.spec_from_file_location("dgpmod", dgp_path)
    d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d
    _W["S"] = {
        "IPW-O-X": _load("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped"),
        "DoublyRobust-O-X": _load("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped"),
        "IPW-O-W": _load("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped"),
        "DoublyRobust-O-W": _load("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")}


def _shap(Xn, sX, sp, block=200):
    Xn = np.asarray(Xn, float).reshape(-1, 1)
    out = np.empty(len(Xn)); dg = np.arange(len(sX))
    for s0 in range(0, len(Xn), block):
        xb = Xn[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        S = dd[:, :, None] + dd[:, None, :]; S = np.where(S > 0, S, 1.0)
        A = (dd[:, None, :] * sp[None, :, None] + dd[:, :, None] * sp[None, None, :]) / S
        A[:, dg, dg] = sp[None, :]
        v = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0
        if ex.any(): v[ex] = sp[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = v
    return out


def run_arm(job):
    seed, n, nte, arm = job
    common = _W["common"]; d = _W["dgp"]; S = _W["S"]; xsup = _W["extract"]; GAM = _W["gammas"]
    obs, _f = d.generate(n, seed); X, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(nte, seed + 1000); Xte = te["X"]; Y1, Y0 = ft["Y1"], ft["Y0"]
    x = np.asarray(X, float).ravel(); xte = np.asarray(Xte, float).ravel()
    ed = np.quantile(x, np.linspace(0, 1, BINS + 1)); ed[0] -= 1e-9; ed[-1] += 1e-9
    ci = np.clip(np.digitize(x, ed) - 1, 0, BINS - 1)
    cte = np.clip(np.digitize(xte, ed) - 1, 0, BINS - 1)
    ctr = np.array([x[ci == j].mean() if (ci == j).any() else 0.5 * (ed[j] + ed[j + 1]) for j in range(BINS)])
    XX = ctr[ci].reshape(-1, 1) if arm == "binned15" else X
    w, _ = common.ipw_weights_from_data(X, T, 2)
    mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
    Dm = common.pairwise_distance_matrix(XX)
    eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=1.0))
    kwarm = {"lipschitz": 3.0} if arm == "free-pi_L3" else ({"linear_policy": True} if arm == "linear" else {})
    out = {}
    for m in METHODS:
        out[m] = {}
        for g in GAM:
            t0 = time.time()
            try:
                if m == "IPW-O-X": r = S[m](XX, T, Y, w, n_arms=2, Gamma=g, discretize=False, **kwarm)
                elif m == "DoublyRobust-O-X": r = S[m](XX, T, Y, w, mu, n_arms=2, Gamma=g, discretize=False, **kwarm)
                elif m == "IPW-O-W": r = S[m](XX, T, Y, w, n_arms=2, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kwarm)
                else: r = S[m](XX, T, Y, w, mu, n_arms=2, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kwarm)
                pi = np.asarray(r.pi[1], float)
                if arm == "binned15":
                    cp = np.array([pi[ci == j].mean() if (ci == j).any() else 0.0 for j in range(BINS)])
                    pe = cp[cte]
                else:
                    sX, spv = xsup(XX, pi)
                    pe = _shap(xte, np.asarray(sX, float), np.asarray(spv, float).ravel())
                out[m]["%g" % g] = {"value": float(np.mean(pe * Y1 + (1 - pe) * Y0)),
                                    "treat": float(np.mean(pe)), "secs": round(time.time() - t0, 1)}
            except Exception as ex:
                print("FAIL %s %s G=%s: %s" % (arm, m, g, ex), flush=True)
                out[m]["%g" % g] = {"value": float("nan")}
    orc = d.oracle_policy(xte)
    print("  %s done (%.0fs)" % (arm, time.time() - t0), flush=True)
    return arm, out, {"oracle": float(np.mean(orc * Y1 + (1 - orc) * Y0)), "never": float(np.mean(Y0))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=4000, dest="nte")
    ap.add_argument("--outdir", default="assets/linear3")
    a = ap.parse_args()
    TL = [(c, p, g, sd) for c, p, g in CAMPS for sd in range(10)]
    camp, dgp, gg, seed = TL[a.task_id]
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    op = outdir / ("%s_s%d.json" % (camp, seed))
    if op.exists():
        print("already done", op); return
    Path("gurobi.env").write_text("Threads 1\n")
    gam = [float(g) for g in gg]
    print("task %d -> %s seed %d | arms=%s | gammas=%s" % (a.task_id, camp, seed, ARMS, gam), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(len(ARMS), initializer=_init,
                                      initargs=(str((ROOT / dgp).resolve()), gam)) as pool:
        res = pool.map(run_arm, [(seed, a.n, a.nte, ar) for ar in ARMS])
    op.write_text(json.dumps({"campaign": camp, "seed": seed, "N_train": a.n,
                              "gammas": ["%g" % g for g in gam], "methods": METHODS, "arms": ARMS,
                              "cells": {ar: o for ar, o, _ in res},
                              "refs": {ar: r for ar, _, r in res}}))
    print("saved %s (%.1f min)" % (op, (time.time() - t0) / 60), flush=True)
    print("DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
