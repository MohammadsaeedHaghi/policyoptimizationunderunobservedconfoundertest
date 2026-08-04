#!/usr/bin/env python3
"""Recompute ONLY the Kallus arm of an existing result cell, after the optimiser fix.

`methods/Kallus/kallus.py` returned a near-uniform softmax under its old defaults (n_iters=12,
n_restarts=2, raw ∇π step, Polyak-average selection). At Γ = 1 that cost 0.9–1.2 units of
normalised value against a 1-second brute force over the two effective parameters; see the
docstring in that file. Every other arm in these cells is unaffected, so re-solving the whole
campaign would be waste — this patches the `kallus` field alone.

The previous value is preserved as `kallus_legacy` (never overwritten once set) so the pre-fix
numbers stay reproducible, per the save-everything rule. Idempotent: re-running recomputes
`kallus` and leaves `kallus_legacy` at the original.
"""
import sys, json, argparse, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semisynth as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="an existing results/*.json cell")
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
    wraw, _ = R._W["common"].ipw_weights_from_data(X, T, 2, normalize=False)

    t0 = time.time()
    r = R._W["kallus"](X, T, Y, wraw, n_arms=2, Gamma=G, maximize=True, seed=int(d["seed"]))
    pe = R._W["kpred"](r.theta, Xte.reshape(-1, 1))[:, 1]
    new = {"value": float(np.mean(pe * Y1t + (1 - pe) * Y0t)),
           "objective": float(r.objective_value),
           "pe_mean": float(pe.mean()), "pe_sd": float(pe.std())}

    if "kallus_legacy" not in d:                 # keep the pre-fix number, once
        d["kallus_legacy"] = d.get("kallus", {})
    old = d.get("kallus", {}).get("%g" % G, {}).get("value", float("nan"))
    d["kallus"] = {"%g" % G: new}
    d["kallus_optimiser"] = {"n_iters": 200, "n_restarts": 6, "eta0": 2.0,
                             "normalize_grad": True, "select": "best", "fixed": "2026-08-04"}
    p.write_text(json.dumps(d))
    print("%-52s  %8.4f -> %8.4f  (%+.4f)  %.0fs"
          % (p.name, old, new["value"], new["value"] - old, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
