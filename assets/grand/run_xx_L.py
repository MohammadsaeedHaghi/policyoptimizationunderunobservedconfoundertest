#!/usr/bin/env python3
"""The X-X (unconfoundedness-assuming) methods swept over the SAME Lipschitz grid as the robust ones.

Why this exists. `run_xx.py` solved the three X-X baselines once, with no `lipschitz` argument --
i.e. only at L = inf. Every robust curve in the reports, meanwhile, is plotted at L = 3 (and its
best L is what the tables quote). So every published O-W-vs-X-X gap was measured across a policy
class mismatch, not just across an uncertainty set, and it flattered us.

It also hid a correctness invariant. At Gamma = 1 the MSM box collapses to the single point {w-hat},
and at c_eps = 1 the tight Wasserstein ball contains w-hat by construction, so IPW-O-X and IPW-O-W
solve literally the same LP as IPW-X-X. Verified here at machine precision (max |pi - pi| = 0).
With X-X pinned at L = inf that identity was invisible on the charts; with a matched L grid the
Gamma = 1 crossing becomes a visible check.

X-X assumes unconfoundedness, so it carries no Gamma -- each (method, L) is one number, plotted as
a flat line across Gamma.

Usage:
  python3 assets/grand/run_xx_L.py --dgp assets/exp_msmbench/dgp.py --n 400 --seeds 5 \
      --out assets/grand/xxL_kmz.json
"""
import sys, json, argparse, importlib.util
import numpy as np

ROOT = "/home1/haghim/code 1.1"
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + "/extensions/Shapley")
import common
from shapley import extract_support

LGRID = [None, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.5]        # identical to the robust campaigns
NTE = 4000
PG = np.linspace(-1, 1, 41)


def LD(rel, fn):
    s = importlib.util.spec_from_file_location(fn, ROOT + "/" + rel)
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


S = {"IPW-X-X": (LD("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped"), "ipw"),
     "DoublyRobust-X-X": (LD("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py",
                             "solve_doublyrobust_x_x_uncapped"), "dr"),
     "Direct-X-X": (LD("methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py",
                       "solve_direct_x_x_uncapped"), "direct")}


def shap(Xn, sX, spv, block=200):
    Xn = np.asarray(Xn, float).reshape(-1, 1); out = np.empty(len(Xn)); dg = np.arange(len(sX))
    for s0 in range(0, len(Xn), block):
        xb = Xn[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        Sm = dd[:, :, None] + dd[:, None, :]; Sm = np.where(Sm > 0, Sm, 1.0)
        A = (dd[:, None, :] * spv[None, :, None] + dd[:, :, None] * spv[None, None, :]) / Sm
        A[:, dg, dg] = spv[None, :]; v = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0
        if ex.any(): v[ex] = spv[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", required=True); ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--out", required=True)
    ap.add_argument("--Ls", default="", help="comma list, e.g. 'inf,3,1'; empty = full LGRID")
    a = ap.parse_args()
    global LGRID
    if a.Ls:
        LGRID = [None if t.strip() == "inf" else float(t) for t in a.Ls.split(",")]
    sp = importlib.util.spec_from_file_location("d", ROOT + "/" + a.dgp)
    d = importlib.util.module_from_spec(sp); sys.modules["d"] = d; sp.loader.exec_module(d)

    Lk = ["inf" if L is None else "%g" % L for L in LGRID]
    vals = {m: {l: [] for l in Lk} for m in S}
    curves = {m: {} for m in S}
    for sd in range(a.seeds):
        obs, _ = d.generate(a.n, sd); X, T, Y = obs["X"], obs["T"], obs["Y"]
        te, ft = d.generate(NTE, sd + 1000); Xte = te["X"]; Y1, Y0 = ft["Y1"], ft["Y0"]
        w, _ = common.ipw_weights_from_data(X, T, 2)
        mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
        for m, (fn, kind) in S.items():
            for L, lk in zip(LGRID, Lk):
                try:
                    if kind == "ipw":   r = fn(X, T, Y, w, n_arms=2, discretize=False, lipschitz=L)
                    elif kind == "dr":  r = fn(X, T, Y, w, mu, n_arms=2, discretize=False, lipschitz=L)
                    else:               r = fn(X, T, Y, n_arms=2, discretize=False, lipschitz=L)
                    sX, spv = extract_support(X, r.pi[1])
                    sX = np.asarray(sX, float).reshape(-1, 1); spv = np.asarray(spv, float).ravel()
                    pe = shap(Xte.ravel(), sX, spv)
                    vals[m][lk].append(float(np.mean(pe * Y1 + (1 - pe) * Y0)))
                    if sd == 0:
                        curves[m][lk] = [round(float(v), 4) for v in shap(PG, sX, spv)]
                except Exception as ex:
                    print("FAIL %s L=%s seed=%d: %s" % (m, lk, sd, str(ex)[:90]), flush=True)
        print("  seed %d done" % sd, flush=True)

    out = {"n": a.n, "seeds": a.seeds, "methods": list(S), "Lgrid": Lk,
           "mean": {m: {l: (round(float(np.mean(v)), 4) if v else None) for l, v in ll.items()}
                    for m, ll in vals.items()},
           "sd": {m: {l: (round(float(np.std(v)), 4) if v else None) for l, v in ll.items()}
                  for m, ll in vals.items()},
           "curves": curves, "policy_grid": [round(float(v), 4) for v in PG]}
    json.dump(out, open(ROOT + "/" + a.out, "w"))
    print("\n%-18s %s" % ("method", " ".join("%8s" % l for l in Lk)))
    for m in S:
        print("%-18s %s" % (m, " ".join("%8.3f" % out["mean"][m][l] if out["mean"][m][l] is not None
                                        else "%8s" % "-" for l in Lk)))
    print("\nsaved " + a.out)


if __name__ == "__main__":
    main()
