#!/usr/bin/env python3
"""A/B test: does the MSM box's CENTER (Hajek-normalised w-hat vs raw 1/e-hat) change results?

Why this exists. The box formula is identical across our solvers, Kallus-Zhou's get_bnds and our
sharp baseline -- verified equal to 1e-14:
    W in [1 + (w_hat - 1)/Gamma,  1 + Gamma (w_hat - 1)]
But the CENTER differs. Our O-X / O-W solvers are fed Hajek-normalised weights (per-arm sum = n),
while the two external references use raw 1/e-hat. The MSM is defined as a bound on the propensity
ODDS RATIO, so raw 1/e-hat is the object the box was derived for; the Hajek version is a
variance-reduction choice. A pre-measurement says the difference is second-order (smallest Gamma
containing the true weights: 4.1 vs 4.1 gstar, 9.9 vs 9.7 KMZ, 38.1 vs 32.5 KZ18; the negative
lower-endpoint pathology fires in 0% of units) -- this script tests it where it matters, on
realised policy value, PAIRED on seed so the comparison is not swamped by seed noise.

Everything else is held fixed: same seeds, same test draws, same Gamma/L/regime cells, same
epsilon rule (epsilon is recomputed from whichever w is in use, since that is what a user
switching the flag would get).

Output: assets/grand/ab_weights.json  +  a paired summary table on stdout.
"""
import json, sys, argparse, importlib.util, time
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
# O-W is EXCLUDED by necessity, not by choice: its Wasserstein radius (common.tight_epsilon)
# solves a transport problem whose marginals balance only when sum_arm w = n exactly -- i.e. it
# REQUIRES the Hajek normalisation, and is infeasible (Gurobi status 3) on raw 1/e-hat. So "raw
# weights for O-W" is not a variant that exists. The box-only methods have no such requirement,
# and that is where the centering question is well posed.
METHODS = ["IPW-O-X", "DoublyRobust-O-X"]
_W = {}


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def _init():
    import common
    _W["common"] = common
    from shapley import extend_with_shapley, extract_support
    _W["extract"] = extract_support
    for tag, (suf, Cap) in (("uncap", ("uncapped", "Uncapped")), ("cap", ("capped", "Capped"))):
        _W[tag] = {
            "IPW-O-X": _load("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf),
            "DoublyRobust-O-X": _load("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf),
            "Hajek-O-X": _load("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (Cap, suf), "solve_hajek_o_x_%s" % suf),
            "IPW-O-W": _load("methods/IPW-O-W/%s/ipw_o_w_%s.py" % (Cap, suf), "solve_ipw_o_w_%s" % suf),
            "DoublyRobust-O-W": _load("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf),
        }


def _shapley_fast(Xnew, sX, sp, block=200):
    if Xnew.ndim == 1: Xnew = Xnew.reshape(-1, 1)
    out = np.empty(Xnew.shape[0]); diag = np.arange(sX.shape[0])
    for s0 in range(0, Xnew.shape[0], block):
        xb = Xnew[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        S = dd[:, :, None] + dd[:, None, :]; S = np.where(S > 0, S, 1.0)
        A = (dd[:, None, :] * sp[None, :, None] + dd[:, :, None] * sp[None, None, :]) / S
        A[:, diag, diag] = sp[None, :]
        v = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0.0
        if ex.any(): v[ex] = sp[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = v
    return out


def one(job):
    """One (campaign, seed, centering) -> value for every method x Gamma x regime at fixed L."""
    camp, dgp_path, seed, centering, gammas, L, caps, n, nte = job
    common = _W["common"]; xsup = _W["extract"]
    sp = importlib.util.spec_from_file_location("dg_%s_%d" % (camp, seed), dgp_path)
    d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); K = d.K

    obs, _f = d.generate(n, seed); X, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(nte, seed + 1000); Xte = te["X"]; Y1t, Y0t = ft["Y1"], ft["Y0"]

    # THE ONLY THING THAT CHANGES between arms of the A/B:
    w, _ = common.ipw_weights_from_data(X, T, K, normalize=(centering == "hajek"))
    wraw, _ = common.ipw_weights_from_data(X, T, K, normalize=False)   # Hajek-O-X always gets raw
    mu = common.outcome_means(X, T, Y, n_arms=K, cross_fit=True)
    eps = None                                    # unused: box-only methods take no radius

    def dep(Xnew, vals):
        sX, spv = xsup(X, vals)
        return _shapley_fast(np.asarray(Xnew, float), np.asarray(sX, float), np.asarray(spv, float).ravel())

    out = {}
    for cap in caps:
        S = _W["cap"] if cap is not None else _W["uncap"]
        kw = {"cap": (1.0, float(cap))} if cap is not None else {}
        rk = "uncap" if cap is None else "cap%g" % (100 * cap)
        out[rk] = {}
        for m in METHODS:
            out[rk][m] = {}
            for g in gammas:
                try:
                    if m == "IPW-O-X": r = S[m](X, T, Y, w, n_arms=K, Gamma=g, discretize=False, lipschitz=L, **kw)
                    elif m == "DoublyRobust-O-X": r = S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, lipschitz=L, **kw)
                    else: r = S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, lipschitz=L, **kw)
                    pe = dep(Xte, r.pi[1])
                    out[rk][m]["%g" % g] = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
                except Exception as ex:
                    print("FAIL %s %s G=%s %s seed%d: %s" % (camp, m, g, centering, seed, ex), flush=True)
                    out[rk][m]["%g" % g] = float("nan")
    print("done %s seed%d %s" % (camp, seed, centering), flush=True)
    return camp, seed, centering, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=4000, dest="nte")
    ap.add_argument("--L", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "ab_weights.json"))
    ap.add_argument("--gammas", default="", help="override; default = matched + a high Gamma per campaign")
    a = ap.parse_args()

    CAMPS = {"gs": ("assets/exp_gstar/dgp_cont.py", [5.0, 12.0]),
             "km": ("assets/exp_msmbench/dgp_g15.py", [4.4817, 12.0]),
             "kz": ("assets/exp_kz18/dgp.py", [4.4817, 24.0])}
    if a.gammas:
        gv = [float(x) for x in a.gammas.split(",")]
        CAMPS = {k: (v[0], gv) for k, v in CAMPS.items()}
    caps = [None, 0.3]
    Path("gurobi.env").write_text("Threads 1\n")
    jobs = [(c, str((ROOT / p).resolve()), sd, cen, gs, a.L, caps, a.n, a.nte)
            for c, (p, gs) in CAMPS.items() for sd in range(a.seeds) for cen in ("hajek", "raw")]
    print("A/B weights: %d jobs (%d campaigns x %d seeds x 2 centerings), L=%g, n=%d"
          % (len(jobs), len(CAMPS), a.seeds, a.L, a.n), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init) as pool:
        res = pool.map(one, jobs)

    D = {}
    for camp, seed, cen, out in res:
        D.setdefault(camp, {}).setdefault(cen, {})[str(seed)] = out
    Path(a.out).write_text(json.dumps({"n": a.n, "L": a.L, "seeds": a.seeds,
                                       "campaigns": {c: CAMPS[c][1] for c in CAMPS}, "data": D}, indent=1))
    print("\nsaved %s (%.1f min)\n" % (a.out, (time.time() - t0) / 60))

    print("PAIRED comparison (raw - hajek), mean over seeds +- sd of the paired difference.")
    print("A difference small relative to its sd means the centering does not matter.\n")
    print("%-4s %-8s %-18s %8s %8s %9s %8s" % ("camp", "regime", "method", "hajek", "raw", "diff", "sd(diff)"))
    for camp in D:
        for rk in D[camp]["hajek"]["0"]:
            for m in METHODS:
                for g in D[camp]["hajek"]["0"][rk][m]:
                    h = np.array([D[camp]["hajek"][str(s)][rk][m][g] for s in range(a.seeds)])
                    r = np.array([D[camp]["raw"][str(s)][rk][m][g] for s in range(a.seeds)])
                    dif = r - h
                    flag = "" if abs(np.mean(dif)) <= 2 * (np.std(dif) / max(np.sqrt(a.seeds), 1) + 1e-12) else "  <-- SHIFT"
                    print("%-4s %-8s %-18s %8.3f %8.3f %+9.3f %8.3f%s"
                          % (camp, rk, "%s G=%s" % (m, g), np.nanmean(h), np.nanmean(r),
                             np.nanmean(dif), np.nanstd(dif), flag))
    print("\nAB_DONE")


if __name__ == "__main__":
    main()
