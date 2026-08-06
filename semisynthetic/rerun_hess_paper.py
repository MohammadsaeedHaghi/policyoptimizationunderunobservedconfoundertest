#!/usr/bin/env python3
"""Recompute the SharpHess arm with the PAPER-EXACT pipeline (their github recipe, end to end)."""
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
    pol = H.hess_paper(X, T, Y, Gamma=G, seed=int(d["seed"]), maximize=True)
    pe = np.clip(H.apply_hess_paper(pol, Xte.reshape(-1, 1)), 0.0, 1.0)
    new = {"value": float(np.mean(pe * Y1t + (1 - pe) * Y0t)),
           "pe_mean": float(pe.mean()), "pe_sd": float(pe.std())}

    if "hess_nn_customopt" not in d:
        d["hess_nn_customopt"] = d.get("hess", {})    # the custom-trainer arm, kept
    old = d.get("hess", {}).get("%g" % G, {}).get("value", float("nan"))
    d["hess"] = {"%g" % G: new}
    d["hess_optimiser"] = {"recipe": "paper-exact", "hidden": [64, 32], "adam_lr": 1e-3,
                           "epochs": 300, "batch": 64, "patience": 10,
                           "split": "disjoint-half", "fixed": "2026-08-05"}
    p.write_text(json.dumps(d))
    print("%-52s  %8.4f -> %8.4f  (%.0fs)"
          % (p.name, old, new["value"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
