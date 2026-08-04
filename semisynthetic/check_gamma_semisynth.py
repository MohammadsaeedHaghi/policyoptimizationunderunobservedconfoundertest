#!/usr/bin/env python3
"""Measure the OPERATIVE Gamma and compare it with the declared e^{2 gamma}.

The paper derives its guidance -- that the best Gamma lies between e^{gamma} and e^{2 gamma} --
from the range of the hidden state alone (|u - u'| <= 2 for u in {-1,+1}). That argument is about
the TRUE nominal propensity. Our pipeline, like theirs, plugs in an ESTIMATED one, so the box
built at the declared Gamma need not contain the true weights at all.

This measures the smallest box around the ESTIMATED propensity that does contain them:

    box at Gamma:  W in [ 1 + (what - 1)/Gamma , 1 + Gamma (what - 1) ],  what = 1/ehat(x)
    smallest containing Gamma per unit:
        (W_true - 1)/(what - 1)   if W_true > what,   else its reciprocal

There is a second reason to expect a gap here beyond estimation error. The assignment depends on
X through lam'X, but the solver's covariate is the CATE index b'X, and lam'X is NOT a function of
b'X. So variation in lam'X that the index cannot see behaves exactly like extra hidden
confounding, on top of u. The operative Gamma should therefore exceed e^{2 gamma}, and by an
amount that does NOT shrink as gamma grows.

Replays run_semisynth.draw() bit-for-bit so the units are the ones the solvers actually saw.
"""
import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path("/home1/haghim/code 1.1")
sys.path.insert(0, str(ROOT))
import common

N_TRAIN = 300
DATASETS = ["bank_marketing", "wine_quality", "german_credit", "credit_default", "adult"]
GAMMAS = [0.0, 1.0, 1.5, 2.0]


def operative(ds, gamma, seed):
    P = np.load(HERE / "prepared" / ("%s_g%g_pop.npz" % (ds, gamma)))
    x, e, Y0, Y1 = P["x"], P["e"], P["Y0"], P["Y1"]
    rng = np.random.default_rng(10_000 + seed)          # identical stream to run_semisynth.draw
    n = len(x)
    T = (rng.uniform(size=n) < e).astype(int)
    _Yo = np.where(T == 1, Y1, Y0)
    perm = rng.permutation(n)
    tr = perm[:N_TRAIN]

    Xtr, Ttr, e_true = x[tr].reshape(-1, 1), T[tr], e[tr]
    ehat = common.propensity_matrix(Xtr, Ttr, 2)[:, 1]   # what the solvers are handed
    W_true = np.where(Ttr == 1, 1.0 / e_true, 1.0 / (1.0 - e_true))
    W_hat = np.where(Ttr == 1, 1.0 / ehat, 1.0 / (1.0 - ehat))
    num, den = W_true - 1.0, W_hat - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(W_true > W_hat, num / den, den / num)
    g = np.maximum(np.abs(g[np.isfinite(g)]), 1.0)
    dec = float(np.exp(2 * gamma))
    return {"p50": float(np.percentile(g, 50)), "p95": float(np.percentile(g, 95)),
            "max": float(g.max()), "frac_outside": float(np.mean(g > dec))}


out = {}
print("Operative Gamma vs the declared e^{2 gamma}  (mean over 10 seeds)\n")
print("%-16s %6s %10s %9s %9s %9s %12s" %
      ("dataset", "gamma", "declared", "median", "p95", "max", "% outside"))
print("-" * 78)
for ds in DATASETS:
    for gm in GAMMAS:
        try:
            r = np.array([[operative(ds, gm, s)[k] for k in ("p50", "p95", "max", "frac_outside")]
                          for s in range(10)]).mean(axis=0)
        except FileNotFoundError:
            continue
        out.setdefault(ds, {})["%g" % gm] = dict(zip(("p50", "p95", "max", "frac_outside"),
                                                     r.tolist()))
        print("%-16s %6.1f %10.3f %9.2f %9.2f %9.2f %11.1f%%"
              % (ds, gm, float(np.exp(2 * gm)), r[0], r[1], r[2], 100 * r[3]))
    print()

json.dump(out, open(HERE / "operative_gamma.json", "w"), indent=1)
print("wrote operative_gamma.json")
