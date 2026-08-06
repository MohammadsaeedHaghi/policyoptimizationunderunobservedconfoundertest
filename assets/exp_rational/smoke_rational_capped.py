#!/usr/bin/env python3
"""The CAPPED arm of the rational-DM DGP: every method under a treatment budget.

The uncapped runner asks who to treat. This asks who to treat when only a `cap` fraction can be,
which is the budgeted-policy setting the capped solvers were written for and the one the Gamma-star
showcase reports at cap 30%.

THREE THINGS CHANGE, and getting them wrong would quietly invalidate the comparison:

1. THE REFERENCES. "Treat everyone" is infeasible under a cap, so it cannot be the ceiling for the
   normalisation. The capped oracle treats the top `cap` fraction ranked by E[CATE | x]; the best
   feasible x-INDEPENDENT policy is either treat-nobody or a random cap-fraction, whichever is
   better. Normalising against the uncapped references would make every method look worse than it
   is, and by a different amount at each cap.

2. DEPLOYMENT RESPECTS THE BUDGET. The capacity is an in-sample constraint, so a policy deployed
   to fresh test points can breach it. Every method -- including the oracle and the naive plug-in --
   is therefore evaluated the same way: rank the test points by the method's own deployed score and
   treat the top `cap` fraction. For a linear objective under a mass budget that greedy rule is
   exact, and applying it uniformly is what keeps the comparison honest.

3. THE SOLVERS. methods/*/Capped/ take `cap` as a per-arm sequence, so cap=(1.0, 0.3) means at most
   30% of units may be assigned to arm 1.
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

from dgp_rational import Cfg, draw
from smoke_rational import _load, _shapley_fast

LGRID = [None, 3.0, 1.0]
CEPS = [1.0, 2.0]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
_W = {}


def _init():
    import common
    from shapley import extract_support
    _W["common"] = common; _W["extract"] = extract_support
    S = {}
    S["IPW-O-X"] = _load("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped")
    S["DoublyRobust-O-X"] = _load("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped")
    S["Hajek-O-X"] = _load("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")
    S["IPW-O-W"] = _load("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped")
    S["DoublyRobust-O-W"] = _load("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped")
    S["Hajek-O-W"] = _load("methods/Hajek-O-W/Capped/hajek_o_w_capped.py", "solve_hajek_o_w_capped")
    S["IPW-X-X"] = _load("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped")
    S["DoublyRobust-X-X"] = _load("methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py", "solve_doublyrobust_x_x_capped")
    S["Direct-X-X"] = _load("methods/Direct-X-X/Capped/direct_x_x_capped.py", "solve_direct_x_x_capped")
    _W["S"] = S


def run_cell(cfg: Cfg, seed: int, n_train: int, n_test: int, cap: float):
    common = _W["common"]; S = _W["S"]
    tr = draw(n_train, 1000 + seed, cfg)            # same draws as the uncapped arm
    te = draw(n_test, 90000 + seed, cfg)
    X = tr["x"].reshape(-1, 1); T = tr["T"]; Yraw = tr["Y"]
    Xte, Y0t, Y1t = te["x"], te["Y0"], te["Y1"]
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)
    n_te = len(Y1t); k = int(np.floor(cap * n_te))

    def eval_capped(score):
        """Treat the top `cap` fraction by the method's own score -- exact for a linear objective
        under a mass budget, and applied identically to every method including the oracle."""
        sc = np.asarray(score, float).ravel()
        pi = np.zeros(n_te)
        if k > 0:
            pi[np.argsort(-sc)[:k]] = 1.0
        return float(np.mean(pi * Y1t + (1 - pi) * Y0t))

    never = float(np.mean(Y0t))
    rand_cap = cap * float(np.mean(Y1t)) + (1 - cap) * never   # best x-independent feasible policy
    refs = {"oracle_cap": eval_capped(te["cate_cond"]), "never": never, "rand_cap": rand_cap,
            "all_uncapped": float(np.mean(Y1t))}
    bc = max(never, rand_cap); sc_ = refs["oracle_cap"] - bc

    w, _ = common.ipw_weights_from_data(X, T, 2)
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X)
    G = float(cfg.Gstar)
    CAP = (1.0, float(cap))

    def dep(vals):
        sX, sp = _W["extract"](X, vals)
        return _shapley_fast(Xte, sX, sp)

    # under a budget the naive analyst ranks by its ESTIMATED effect and takes the top cap
    # fraction -- thresholding at zero first would throw away the ranking the budget needs
    naive = eval_capped(dep(mu[:, 1] - mu[:, 0]))

    grid = {}
    for m in OX + OW:
        grid[m] = {}
        for ce in CEPS:
            if m in OX and ce != CEPS[0]:
                continue
            eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=ce))
            ck = "%g" % ce; grid[m][ck] = {}
            for L, lk in zip(LGRID, ["inf" if L is None else "%g" % L for L in LGRID]):
                try:
                    kw = dict(n_arms=2, Gamma=G, cap=CAP, discretize=False, lipschitz=L)
                    if m == "IPW-O-X":            r = S[m](X, T, Y, w, **kw)
                    elif m == "DoublyRobust-O-X": r = S[m](X, T, Y, w, mu, **kw)
                    elif m == "Hajek-O-X":        r = S[m](X, T, Y, wraw, maximize=True, **kw)
                    else:
                        kw.update(zscore=False, epsilon=eps)
                        if m == "IPW-O-W": r = S[m](X, T, Y, w, **kw)
                        elif m == "DoublyRobust-O-W": r = S[m](X, T, Y, w, mu, **kw)
                        else: r = S[m](X, T, Y, w, maximize=True, **kw)
                    grid[m][ck][lk] = eval_capped(dep(r.pi[1]))
                except Exception as ex:
                    print("FAIL %s ce=%s L=%s: %s" % (m, ck, lk, str(ex)[:90]), flush=True)

    xx = {}
    for m in XX:
        xx[m] = {}
        for L, lk in zip(LGRID, ["inf" if L is None else "%g" % L for L in LGRID]):
            try:
                kw = dict(n_arms=2, cap=CAP, discretize=False, lipschitz=L)
                if m == "IPW-X-X": r = S[m](X, T, Y, w, **kw)
                elif m == "DoublyRobust-X-X": r = S[m](X, T, Y, w, mu, **kw)
                else: r = S[m](X, T, Y, **kw)
                xx[m][lk] = eval_capped(dep(r.pi[1]))
            except Exception as ex:
                print("FAIL %s L=%s: %s" % (m, lk, str(ex)[:90]), flush=True)

    # the two published baselines, ranked and truncated by the same budget rule as everything else
    bl = {}
    try:
        fit_kallus = _load("methods/Kallus/kallus.py", "fit_kallus")
        predict_kallus = _load("methods/Kallus/kallus.py", "predict_kallus")
        r = fit_kallus(X, T, Y, wraw, n_arms=2, Gamma=G, maximize=True, seed=seed)
        bl["Kallus"] = eval_capped(predict_kallus(r.theta, Xte.reshape(-1, 1))[:, 1])
    except Exception as ex:
        print("FAIL Kallus: %s" % str(ex)[:90], flush=True)
    try:
        import sharp_hess as H
        for tag, nu in (("SharpHess", "nn"), ("SharpHess-kNN", "knn")):
            kw = dict(Gamma=G, n_folds=2, maximize=True, seed=seed, nuisance=nu)
            if nu == "knn": kw["k"] = 15
            sc2 = H.fit_scores(X, T, Y, **kw)
            th = H.learn_policy_parametric(X, sc2, seed=seed, maximize=True)
            bl[tag] = eval_capped(H.apply_policy(th, Xte.reshape(-1, 1)))
    except Exception as ex:
        print("FAIL SharpHess: %s" % str(ex)[:90], flush=True)

    return {"refs": refs, "bc": bc, "sc": sc_, "cap": cap, "naive": naive,
            "grid": grid, "xx": xx, "baselines": bl}


def main():
    ap = argparse.ArgumentParser()
    for kk, v in Cfg().as_dict().items():
        # Cfg now carries a STRING field (`shape`), so the type has to follow the default
        # rather than being float for everything.
        ap.add_argument("--" + kk, type=type(v), default=v)
    ap.add_argument("--cap", type=float, default=0.3)
    ap.add_argument("--single-seed", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--tag", default="cfg")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = Cfg(**{kk: getattr(a, kk) for kk in Cfg().as_dict()})

    _init()
    t0 = time.time()
    cell = run_cell(cfg, a.single_seed, a.n_train, a.n_test, a.cap)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tag": a.tag, "cfg": cfg.as_dict(), "cap": a.cap, "seed": a.single_seed,
               "n_train": a.n_train, "cell": cell}, open(a.out, "w"))
    ow = max((v for c in cell["grid"].get("IPW-O-W", {}).values() for v in c.values()
              if v == v), default=float("nan"))
    print("  %s cap=%.2f seed %d: headroom %.3f, IPW-O-W %+.3f  (%.0fs)"
          % (a.tag, a.cap, a.single_seed, cell["sc"],
             (ow - cell["bc"]) / cell["sc"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
