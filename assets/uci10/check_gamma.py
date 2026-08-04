#!/usr/bin/env python3
"""Is Gamma = 4 actually the right box, given the learner gets an ESTIMATED propensity?

Lambda = 4 is exact with respect to the TRUE nominal propensity e(x) = E[e(x,S) | x]: the odds
ratio between S = +1 and S = -1 is 4 at every x, by construction and with no clipping. But the
solvers never see e(x). They estimate ehat(x) from the 300 training rows, and the MSM box is
built around 1/ehat, not around 1/e. So the box at Gamma = 4 need not contain the true weights.

This measures the gap directly. For each training unit the MSM box is
    W  in  [ 1 + (what - 1)/Gamma ,  1 + Gamma (what - 1) ],   what = 1 / ehat(x_i)
and the true weight is W_true = 1 / e(x_i, S_i). The smallest Gamma that contains it is
    (W_true - 1) / (what - 1)   if W_true > what,    else   (what - 1) / (W_true - 1)
and the OPERATIVE Gamma for the dataset is the max over units (plus the 95th percentile, which
is the more useful number -- the max is set by whichever single unit has the worst ehat).

Usage: python3 assets/uci10/check_gamma.py
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
import common

N_TRAIN = 300
NAMES = ["adult", "bank_marketing", "credit_default", "online_shoppers", "mushroom",
         "wine_quality", "spambase", "support2", "aids_clinical", "communities_crime"]


def operative_gamma(name, seed):
    """Replay run_uci.draw() exactly, then compare the true weights to the estimated box."""
    P = np.load(HERE / "data" / (name + "_pop.npz"))
    x, e, m0, m1 = P["x"], P["e"], P["mu0"], P["mu1"]
    binary = bool(P["kind"][0] > 0.5)
    n = len(x)
    rng = np.random.default_rng(10_000 + seed)          # identical stream to the runner
    T = (rng.uniform(size=n) < e).astype(int)
    if binary:
        Y0 = (rng.uniform(size=n) < m0).astype(float)
        Y1 = (rng.uniform(size=n) < m1).astype(float)
    else:
        Y0 = m0 + rng.normal(0, 0.3, n); Y1 = m1 + rng.normal(0, 0.3, n)
    perm = rng.permutation(n); tr = perm[:N_TRAIN]

    Xtr, Ttr = x[tr].reshape(-1, 1), T[tr]
    e_true = e[tr]                                       # true e(x, S) -- depends on hidden S
    Phat = common.propensity_matrix(Xtr, Ttr, 2)         # what the solvers actually get: ehat(x)
    e_hat = Phat[:, 1]

    # weight of the arm each unit was actually observed in
    W_true = np.where(Ttr == 1, 1.0 / e_true, 1.0 / (1.0 - e_true))
    W_hat = np.where(Ttr == 1, 1.0 / e_hat, 1.0 / (1.0 - e_hat))

    num, den = W_true - 1.0, W_hat - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(W_true > W_hat, num / den, den / num)
    g = np.abs(g[np.isfinite(g)])
    g = np.maximum(g, 1.0)
    return {"max": float(g.max()), "p95": float(np.percentile(g, 95)),
            "p50": float(np.percentile(g, 50)),
            "frac_outside_4": float(np.mean(g > 4.0))}


def main():
    print("\nOperative Gamma -- smallest box around the ESTIMATED propensity that still contains")
    print("the TRUE weights.  Declared Lambda = 4 (exact w.r.t. the true nominal propensity).\n")
    print("%-19s %8s %8s %8s %14s" % ("dataset", "median", "p95", "max", "% units > 4"))
    print("-" * 62)
    out = {}
    for nm in NAMES:
        r = np.array([[operative_gamma(nm, s)[k] for k in ("p50", "p95", "max", "frac_outside_4")]
                      for s in range(10)]).mean(axis=0)
        out[nm] = dict(zip(("p50", "p95", "max", "frac_outside_4"), r.tolist()))
        print("%-19s %8.2f %8.2f %8.2f %13.1f%%" % (nm, r[0], r[1], r[2], 100 * r[3]))
    json.dump(out, open(HERE / "operative_gamma.json", "w"), indent=1)
    print("\n(mean over the same 10 seeds the campaign used; saved operative_gamma.json)\n")


if __name__ == "__main__":
    main()
