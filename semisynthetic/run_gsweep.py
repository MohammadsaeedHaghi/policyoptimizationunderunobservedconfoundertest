#!/usr/bin/env python3
"""Gamma MIS-SPECIFICATION sweep on the semi-synthetic campaign (the 5 UCI datasets).

The main campaign solves only at the matched Gamma = e^{2 gamma}. This holds the DGP at the
confounded strength gamma = 1 (true Gamma* = 7.389) and varies the Gamma the SOLVERS assume:

    Gamma in {1, 2, 7.389 (matched), 15, 50}

i.e. the method's robustness hyperparameter swept from "no robustness" to ~7x the truth --
the same mis-specification axis the KMZ tab now reports. Per Gamma: the O-X and O-W families
over L in {inf, 3, 1} at c_eps = 1 (the campaign's best cell was c_eps = 1 almost everywhere),
plus the two authors'-code baselines. X-X and naive are Gamma-free and already stored in the
main campaign, so they are not re-solved.

Scope: 5 datasets x 5 DGP draws x 5 split seeds (half the campaign's seed count -- this is a
sweep, not the headline), one job per (dataset, draw, seed) cell. Draws are IDENTICAL to the
main campaign's (same R.draw), so cells are directly comparable.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semisynth as R

GGRID = [1.0, 2.0, float(np.exp(2.0)), 15.0, 50.0]
LGRID = [None, 3.0, 1.0]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dgp-seed", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    npz = str(HERE / "prepared" / ("%s_g1_d%d_pop.npz" % (a.data, a.dgp_seed)))
    R._init(npz)
    common = R._W["common"]; S = R._W["S"]
    d = R.draw(a.seed)
    X, T, Yraw, Xte = d["Xtr"], d["Ttr"], d["Ytr"], d["Xte"]
    Y0t, Y1t = d["Y0te"], d["Y1te"]
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)

    orc = (d["cate_te"] > 0).astype(float)
    refs = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
            "never_treat": float(np.mean(Y0t)), "all_treat": float(np.mean(Y1t))}

    w, _ = common.ipw_weights_from_data(X, T, 2)
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X)
    eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=1.0))
    Lk = ["inf" if L is None else "%g" % L for L in LGRID]

    def val(vals):
        sX, sp = R._W["extract"](X, vals)
        pe = R._shapley_fast(Xte.reshape(-1, 1), sX, sp)
        return float(np.mean(pe * Y1t + (1 - pe) * Y0t))

    t0 = time.time()
    grid = {}
    for G in GGRID:
        gk = "%g" % G
        grid[gk] = {}
        for m in OX + OW:
            grid[gk][m] = {}
            for L, lk in zip(LGRID, Lk):
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
                    grid[gk][m][lk] = val(r.pi[1])
                except Exception as ex:
                    print("FAIL %s G=%s L=%s: %s" % (m, gk, lk, str(ex)[:80]), flush=True)

    # the two authors'-code baselines at each Gamma
    bl = {}
    import importlib.util as _iu
    _sp = _iu.spec_from_file_location("kal_g", str(Path("/home1/haghim/code 1.1/methods/Kallus/kallus.py")))
    _km = _iu.module_from_spec(_sp); sys.modules["kal_g"] = _km; _sp.loader.exec_module(_km)
    H = R._W["hess"]
    for G in GGRID:
        gk = "%g" % G
        bl[gk] = {}
        try:
            rk = _km.fit_kallus_paper(X, T, Y, wraw, n_arms=2, Gamma=G, maximize=True, seed=a.seed)
            pe = _km.predict_kallus_paper(rk.theta, Xte.reshape(-1, 1))[:, 1]
            bl[gk]["Kallus"] = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
        except Exception as ex:
            print("FAIL Kallus G=%s: %s" % (gk, str(ex)[:80]), flush=True)
        try:
            pol = H.hess_paper(X, T, Y, Gamma=G, seed=a.seed, maximize=True)
            pe = np.clip(H.apply_hess_paper(pol, Xte.reshape(-1, 1)), 0.0, 1.0)
            bl[gk]["SharpHess"] = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
        except Exception as ex:
            print("FAIL Hess G=%s: %s" % (gk, str(ex)[:80]), flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"dataset": a.data, "dgp_gamma": 1.0, "dgp_seed": a.dgp_seed, "seed": a.seed,
               "ggrid": ["%g" % g for g in GGRID], "matched": "%g" % float(np.exp(2.0)),
               "ceps": 1.0, "lipschitz": Lk, "refs": refs, "grid": grid, "baselines": bl},
              open(a.out, "w"))
    print("saved %s (%.0fs)" % (a.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
