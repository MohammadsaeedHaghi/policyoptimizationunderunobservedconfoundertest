"""Per-seed FULL-GRID runner: sweeps Gamma x Lipschitz x c_epsilon x cap for ONE seed.

Motivation (2026-07-30). Two facts made the old (seed-parallel, single-eps, single-cap) runner
the wrong decomposition:

  1. Every campaign's best cell sat at Gamma = 8, the TOP of the old grid -- the sweep was
     truncated, not converged. The grid here runs well past the DGP's Gamma*.
  2. Gamma* is exact only w.r.t. the DGP's TRUE nominal propensity. Our pipeline plugs in an
     ESTIMATED nominal propensity, so the box built at Gamma* need not contain the true
     weights at all. Measured minimum Gamma that does contain them, at n = 200:
        gstar 3.4-4.0 (Gamma* = 5, comfortable)   KMZ 9.9 (Gamma* = 4.48)   KZ18 33.7 (!)
     So "matched Gamma" is a property of the DGP, not of the estimator, and the operative
     Gamma has to be swept -- which is exactly what this runner does.

One job = one seed = the whole grid, parallelised over (c_eps, cap) combinations. Saves the
full surface AND every raw pointwise policy, per the save-everything rule.

Usage:
  python3 assets/run_grid_seed.py --dgp assets/exp_gstar/dgp_cont.py --out OUT.json --seed 3 \
      --n 200 --n-test 4000 --gammas 1,2,3,5,8,12,16,24,32 --ceps 0.5,1.0,2.0 \
      --caps none,0.3,0.4,0.5 --workers 6 --deploy shapley
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "KNN"))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))

LGRID = [None, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.5]
METHODS = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
PGRID = np.linspace(-1.0, 1.0, 41)
_W = {}


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def _init(dgp_path, gammas):
    """Load BOTH solver families once; the per-job cap decides which is used."""
    import common
    _W["common"] = common
    from shapley import extend_with_shapley, extract_support
    from knn import extend_with_knn
    _W["extract"] = extract_support; _W["knn"] = extend_with_knn
    sp = importlib.util.spec_from_file_location("dgpmod", dgp_path)
    d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d
    _W["gammas"] = list(gammas)
    for tag, (suf, Cap) in (("uncap", ("uncapped", "Uncapped")), ("cap", ("capped", "Capped"))):
        S = {}
        S["IPW-O-X"] = _load("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf)
        S["DoublyRobust-O-X"] = _load("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf)
        S["Hajek-O-X"] = _load("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (Cap, suf), "solve_hajek_o_x_%s" % suf)
        S["IPW-O-W"] = _load("methods/IPW-O-W/%s/ipw_o_w_%s.py" % (Cap, suf), "solve_ipw_o_w_%s" % suf)
        S["DoublyRobust-O-W"] = _load("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf)
        _W[tag] = S


def _shapley_fast(Xnew, sX, sp, block=200):
    if Xnew.ndim == 1: Xnew = Xnew.reshape(-1, 1)
    out = np.empty(Xnew.shape[0]); diag = np.arange(sX.shape[0])
    for s0 in range(0, Xnew.shape[0], block):
        xb = Xnew[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        S = dd[:, :, None] + dd[:, None, :]; S = np.where(S > 0, S, 1.0)
        A = (dd[:, None, :] * sp[None, :, None] + dd[:, :, None] * sp[None, None, :]) / S
        A[:, diag, diag] = sp[None, :]
        val = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0.0
        if ex.any(): val[ex] = sp[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = val
    return out


def solve_cell(job):
    """One (c_eps, cap) combination: sweep the whole Gamma x L grid for it."""
    seed, N, Nte, ceps, cap, deploy, k = job
    common = _W["common"]; d = _W["dgp"]; K = d.K; GAMMAS = _W["gammas"]
    S = _W["cap"] if cap is not None else _W["uncap"]
    kw0 = {"cap": (1.0, float(cap))} if cap is not None else {}
    if deploy == "shapley":
        xsup = _W["extract"]
        def dep(Xnew, Xs, vals):
            sX, sp = xsup(Xs, vals)
            return _shapley_fast(np.asarray(Xnew, float), np.asarray(sX, float), np.asarray(sp, float).ravel())
    else:
        knn = _W["knn"]
        def dep(Xnew, Xs, vals): return knn(Xnew, Xs, vals, k=k)

    obs, full = d.generate(N, seed); X, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(Nte, seed + 1000); Xte = te["X"]; Y1t, Y0t = ft["Y1"], ft["Y0"]
    w, _ = common.ipw_weights_from_data(X, T, K)
    wraw, _ = common.ipw_weights_from_data(X, T, K, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X)
    eps = tuple(common.tight_epsilon(Dm, T, w, K, is_distance=True, c_eps=ceps))

    Gk = ["%g" % g for g in GAMMAS]
    Lk = ["inf" if L is None else ("%g" % L) for L in LGRID]
    grid = {m: {g: {l: float("nan") for l in Lk} for g in Gk} for m in METHODS}
    pol = {m: {g: {} for g in Gk} for m in METHODS}
    sup = {m: {g: {} for g in Gk} for m in METHODS}
    Pg = PGRID.reshape(-1, 1)

    def call(m, g, L):
        if m == "IPW-O-X": return S[m](X, T, Y, w, n_arms=K, Gamma=g, discretize=False, lipschitz=L, **kw0)
        if m == "DoublyRobust-O-X": return S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, lipschitz=L, **kw0)
        if m == "Hajek-O-X": return S[m](X, T, Y, wraw, n_arms=K, Gamma=g, maximize=True, discretize=False, lipschitz=L, **kw0)
        if m == "IPW-O-W": return S[m](X, T, Y, w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=L, **kw0)
        return S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=L, **kw0)

    for m in METHODS:
        for g in GAMMAS:
            for L, lk in zip(LGRID, Lk):
                try:
                    res = call(m, g, L)
                    pe = dep(Xte, X, res.pi[1])
                    grid[m]["%g" % g][lk] = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
                    pol[m]["%g" % g][lk] = [round(float(v), 4) for v in dep(Pg, X, res.pi[1])]
                    sup[m]["%g" % g][lk] = [round(float(v), 4) for v in res.pi[1]]
                except Exception as ex:
                    print("FAIL %s G=%s L=%s ceps=%s cap=%s seed=%d: %s" % (m, g, lk, ceps, cap, seed, ex), flush=True)

    orc = d.oracle_policy(Xte.ravel())
    nv = (mu[:, 1] - mu[:, 0] > 0).astype(float); nve = dep(Xte, X, nv)
    refs = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
            "never_treat": float(np.mean(Y0t)), "all_treat": float(np.mean(Y1t)),
            "naive_dr": float(np.mean(nve * Y1t + (1 - nve) * Y0t)),
            "epsilon": [float(e) for e in eps]}
    pol["_refs"] = {"oracle": [round(float(v), 4) for v in d.oracle_policy(PGRID)],
                    "naive_dr": [round(float(v), 4) for v in dep(Pg, X, nv)]}
    sup["_X"] = [round(float(v), 5) for v in np.asarray(X).ravel()]
    sup["_naive_dr"] = [round(float(v), 4) for v in nv]
    key = "ce%g_%s" % (ceps, "uncap" if cap is None else "cap%g" % (100 * cap))
    print("done %s (%d cells)" % (key, sum(1 for m in METHODS for g in Gk for l in Lk
                                           if grid[m][g][l] == grid[m][g][l])), flush=True)
    return key, grid, refs, pol, sup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=4000, dest="ntest")
    ap.add_argument("--gammas", default="1,2,3,5,8,12,16,24,32")
    ap.add_argument("--ceps", default="0.5,1.0,2.0")
    ap.add_argument("--caps", default="none,0.3,0.4,0.5")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--deploy", default="shapley", choices=["knn", "shapley"])
    a = ap.parse_args()

    gammas = [float(v) for v in a.gammas.split(",")]
    ceps = [float(v) for v in a.ceps.split(",")]
    caps = [None if c.strip().lower() in ("none", "uncap", "") else float(c) for c in a.caps.split(",")]
    Path("gurobi.env").write_text("Threads 1\n")
    jobs = [(a.seed, a.n, a.ntest, ce, cp, a.deploy, a.k) for ce in ceps for cp in caps]
    print("GRID seed=%d n=%d gammas=%s ceps=%s caps=%s -> %d cells x %d L x %d methods, workers=%d dgp=%s"
          % (a.seed, a.n, gammas, ceps, caps, len(jobs), len(LGRID), len(METHODS), a.workers, a.dgp), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                      initargs=(str(Path(a.dgp).resolve()), gammas)) as pool:
        res = pool.map(solve_cell, jobs)

    Gk = ["%g" % g for g in gammas]; Lk = ["inf" if L is None else ("%g" % L) for L in LGRID]
    out = {"seed": a.seed, "N_train": a.n, "N_test": a.ntest, "deploy": a.deploy,
           "methods": METHODS, "gammas": Gk, "Lgrid": Lk,
           "ceps": ceps, "caps": ["uncap" if c is None else "%g" % c for c in caps],
           "dgp": str(Path(a.dgp).resolve()), "policy_grid": PGRID.tolist(),
           "cells": {}, "refs": {}, "policies": {}, "policies_support": {}}
    for key, grid, refs, pol, sup in res:
        out["cells"][key] = grid; out["refs"][key] = refs
        out["policies"][key] = pol; out["policies_support"][key] = sup
    Path(a.out).write_text(json.dumps(out))
    print("saved %s (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)

    # per-key best cell, for the log
    for key in out["cells"]:
        best = max(((m, g, l, out["cells"][key][m][g][l]) for m in METHODS for g in Gk for l in Lk
                    if out["cells"][key][m][g][l] == out["cells"][key][m][g][l]),
                   key=lambda t: t[3], default=None)
        if best:
            print("  %-16s best %-18s %.3f at G=%s L=%s (oracle %.3f naive %.3f)"
                  % (key, best[0], best[3], best[1], best[2],
                     out["refs"][key]["oracle"], out["refs"][key]["naive_dr"]), flush=True)
    print("DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
