#!/usr/bin/env python3
"""Gamma sweep for the RECOMMENDED rational-DM config (nocouple: a=2, alpha=0, delta=1,
beta0=2, Gamma*=5) -- the data behind a KMZ-style "across Gamma" tab.

The DGP is FIXED (its true Gamma* stays 5); what varies is the Gamma the SOLVERS assume:

    Gamma in {1, 2, 3, 5 (matched), 8, 15, 50}

Per (seed, gamma) task:
  - O-X (IPW/DR/Hajek) over the KMZ L grid {inf,10,5,3,2,1.5,1,0.5} at c_eps = 1.0 only
    (the odds box has no Wasserstein radius -- epsilon-invariance verified on KMZ).
  - O-W (IPW/DR/Hajek) over the same L grid at c_eps in {1.0, 1.5, 2.0}.
  - Kallus (paper port) and SharpHess (authors' recipe) at this Gamma.
  - On the gamma=1 task only: the Gamma-free extras -- X-X (IPW/DR/Direct) over the SAME
    full L grid, and the naive plug-in.

Draws are EXACTLY the confirmation campaign's (draw(n, 1000+seed) / draw(n_test, 90000+seed)),
so every cell is paired with the existing nocouple results. Outcome centring, Shapley
deployment, and solver signatures are inherited verbatim from smoke_rational.
"""
from __future__ import annotations

import sys, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/home1/haghim/code 1.1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
sys.path.insert(0, str(ROOT / "methods" / "SharpHess"))
sys.path.insert(0, str(HERE))

from dgp_rational import Cfg, draw, check
import smoke_rational as SR

GGRID = [1.0, 2.0, 3.0, 5.0, 8.0, 15.0, 50.0]
LGRID = [None, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.5]
CEPS = [1.0, 1.5, 2.0]
LK = ["inf" if L is None else "%g" % L for L in LGRID]


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gamma", type=float, required=True)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cfg = Cfg(a=2.0, alpha=0.0, delta=1.0, beta0=2.0)   # nocouple, Gamma* = 5
    ck = check(cfg)
    assert ck["gamma_exact"] and ck["rational"], "design invariants fail"

    SR._init()
    common = SR._W["common"]; S = SR._W["S"]
    import sharp_hess as H
    fit_kallus = _load("methods/Kallus/kallus.py", "fit_kallus_paper")
    predict_kallus = _load("methods/Kallus/kallus.py", "predict_kallus_paper")

    sd = a.seed
    tr = draw(a.n_train, 1000 + sd, cfg)
    te = draw(a.n_test, 90000 + sd, cfg)
    X = tr["x"].reshape(-1, 1); T = tr["T"]; Yraw = tr["Y"]
    Xte, Y0t, Y1t = te["x"], te["Y0"], te["Y1"]
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)

    orc = (te["cate_cond"] > 0).astype(float)
    refs = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
            "never": float(np.mean(Y0t)), "all": float(np.mean(Y1t))}
    bc = max(refs["never"], refs["all"]); sc = refs["oracle"] - bc

    w, _ = common.ipw_weights_from_data(X, T, 2)
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X)
    G = float(a.gamma)

    def val(vals):
        pe = SR._shapley_fast(Xte, *SR._W["extract"](X, vals))
        return float(np.mean(pe * Y1t + (1 - pe) * Y0t))

    t0 = time.time()
    grid = {}
    for m in SR.OX + SR.OW:
        grid[m] = {}
        for ce in (CEPS if m in SR.OW else CEPS[:1]):
            eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=ce))
            ckey = "%g" % ce; grid[m][ckey] = {}
            for L, lk in zip(LGRID, LK):
                try:
                    kw = dict(n_arms=2, Gamma=G, discretize=False, lipschitz=L)
                    if m == "IPW-O-X":            r = S[m](X, T, Y, w, **kw)
                    elif m == "DoublyRobust-O-X": r = S[m](X, T, Y, w, mu, **kw)
                    elif m == "Hajek-O-X":        r = S[m](X, T, Y, wraw, maximize=True, **kw)
                    else:
                        kw.update(zscore=False, epsilon=eps)
                        if m == "IPW-O-W": r = S[m](X, T, Y, w, **kw)
                        elif m == "DoublyRobust-O-W": r = S[m](X, T, Y, w, mu, **kw)
                        else: r = S[m](X, T, Y, w, maximize=True, **kw)
                    grid[m][ckey][lk] = val(r.pi[1])
                except Exception as ex:
                    print("FAIL %s ce=%s L=%s: %s" % (m, ckey, lk, str(ex)[:80]), flush=True)

    bl = {}
    try:
        r = fit_kallus(X, T, Y, wraw, n_arms=2, Gamma=G, maximize=True, seed=sd)
        pe = np.clip(predict_kallus(r.theta, Xte.reshape(-1, 1))[:, 1], 0.0, 1.0)
        bl["Kallus"] = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
    except Exception as ex:
        print("FAIL Kallus: %s" % str(ex)[:120], flush=True)
    try:
        pol = H.hess_paper(X, T, Y, Gamma=G, seed=sd, maximize=True)
        pe = np.clip(H.apply_hess_paper(pol, Xte.reshape(-1, 1)), 0.0, 1.0)
        bl["SharpHess"] = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
    except Exception as ex:
        print("FAIL SharpHess: %s" % str(ex)[:120], flush=True)

    out = {"tag": "nocouple", "cfg": cfg.as_dict(), "seed": sd, "gamma": "%g" % G,
           "matched": "5", "n_train": a.n_train, "Lgrid": LK,
           "refs": refs, "bc": bc, "sc": sc, "grid": grid, "baselines": bl}

    if G == 1.0:      # the Gamma-free extras, once per seed
        xx = {}
        for m in SR.XX:
            xx[m] = {}
            for L, lk in zip(LGRID, LK):
                try:
                    if m == "IPW-X-X":
                        r = S[m](X, T, Y, w, n_arms=2, discretize=False, lipschitz=L)
                    elif m == "DoublyRobust-X-X":
                        r = S[m](X, T, Y, w, mu, n_arms=2, discretize=False, lipschitz=L)
                    else:
                        r = S[m](X, T, Y, n_arms=2, discretize=False, lipschitz=L)
                    xx[m][lk] = val(r.pi[1])
                except Exception as ex:
                    print("FAIL %s L=%s: %s" % (m, lk, str(ex)[:80]), flush=True)
        out["xx"] = xx
        out["naive"] = val((mu[:, 1] - mu[:, 0] > 0).astype(float))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("saved %s (%.0fs)" % (a.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
