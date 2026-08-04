#!/usr/bin/env python3
"""Recompute ONLY the SharpHess arm of an existing result cell, after the optimiser fix.

`methods/SharpHess/sharp_hess.py` used a raw grad-pi step with a fixed lr from a fixed small init,
the same defect found in the Kallus baseline; measured against a brute force over its own policy
class it lost up to 0.96 normalised value. The campaign compounded it by calling the learner with
n_iter=300, lr=0.05, restarts=3 -- weaker than the module's own defaults -- and with k=15.

The Eq.-15 SCORE formulas were verified correct throughout (test_gamma1_is_aipw passes at 9e-16),
so only the policy-optimisation step changes. Previous value preserved as `hess_legacy`.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semisynth as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    a = ap.parse_args()
    p = Path(a.file)
    d = json.loads(p.read_text())

    npz = str(HERE / "prepared" / ("%s_g%g_d%d_pop.npz"
                                   % (d["dataset"], d["gamma"], d.get("dgp_seed", 0))))
    R._init(npz)
    G = float(d["Gamma"])
    dd = R.draw(int(d["seed"]))
    X, T, Yraw, Xte = dd["Xtr"], dd["Ttr"], dd["Ytr"], dd["Xte"]
    Y0t, Y1t = dd["Y0te"], dd["Y1te"]
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)
    H = R._W["hess"]

    t0 = time.time()
    sc = H.fit_scores(X, T, Y, Gamma=G, k=50, n_folds=2, maximize=True, seed=int(d["seed"]))
    th = H.learn_policy_parametric(X, sc, seed=int(d["seed"]), maximize=True)
    pe = H.apply_policy(th, Xte.reshape(-1, 1))
    new = {"value": float(np.mean(pe * Y1t + (1 - pe) * Y0t)),
           "pe_mean": float(pe.mean()), "pe_sd": float(pe.std())}

    if "hess_legacy" not in d:
        d["hess_legacy"] = d.get("hess", {})
    old = d.get("hess", {}).get("%g" % G, {}).get("value", float("nan"))
    d["hess"] = {"%g" % G: new}
    d["hess_optimiser"] = {"k": 50, "n_iter": 800, "lr": 2.0, "restarts": 6,
                           "normalize_grad": True, "select": "best", "fixed": "2026-08-04"}
    p.write_text(json.dumps(d))
    print("%-52s  %8.4f -> %8.4f  (%+.4f)  %.0fs"
          % (p.name, old, new["value"], new["value"] - old, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
