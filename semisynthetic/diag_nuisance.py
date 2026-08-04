#!/usr/bin/env python3
"""Are the paper's NEURAL nuisances actually more accurate here than k-NN, or just different?

Switching to the paper's Table-5 neural instantiation made SharpHess much better at gamma=0
(+0.421 -> +0.742) and much worse at every confounded gamma (gamma=1: +0.593 -> +0.166). Before
reporting that, check the nuisances against GROUND TRUTH rather than against each other.

We have the full population (20k rows) behind every cell, so the true conditional nuisances are
computable by taking a large local neighbourhood in the POPULATION around each training x:

    e(x)            true propensity, stored exactly in the npz
    F^-1_{x,a}(a+)  population conditional quantile at alpha^+
    mu^+(a,x)       population E[Y 1{Y <= F^-1}]        (lower-tail truncated mean)
    mubar^+(a,x)    population E[Y 1{Y >= F^-1}]

Reports RMSE of each estimator against those. The suspicion worth testing: Y0 is in {-1,+1}, so
the outcome is near two-valued and its conditional quantile is a STEP in x -- which a smooth MLP
smears and an empirical k-NN quantile reproduces exactly. If that is what is happening, the k-NN
advantage at confounded gamma is a property of this DGP's binary outcome, not a defect in either.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semisynth as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="german_credit")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dgp-seed", type=int, default=0)
    ap.add_argument("--npop", type=int, default=400, help="population neighbours per reference point")
    a = ap.parse_args()

    npz = str(HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, a.gamma, a.dgp_seed)))
    R._init(npz)
    H = R._W["hess"]
    G = float(np.exp(2 * a.gamma))
    a_plus = G / (1.0 + G)
    d = R.draw(a.seed)
    X, T, Yraw = d["Xtr"], d["Ttr"], d["Ytr"]
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)
    ym, ys = float(np.mean(Yraw)), (float(np.std(Yraw)) or 1.0)

    P = R._W["pop"]
    xp, ep, Y0p, Y1p = P["x"].ravel(), P["e"].ravel(), P["Y0"], P["Y1"]
    # the population's own observed outcome under the SAME assignment law, on the training scale
    rng = np.random.default_rng(777)
    Tp = (rng.uniform(size=len(xp)) < ep).astype(int)
    Yp = np.where(Tp == 1, Y1p, Y0p)
    Yp = (Yp - ym) / ys

    xtr = X.ravel()
    nq = int(a.npop)
    e_true = np.zeros(len(xtr)); q_true = np.zeros((len(xtr), 2))
    mlo_true = np.zeros((len(xtr), 2)); mhi_true = np.zeros((len(xtr), 2))
    order = np.argsort(xp)
    xs, es, Ts, Ys = xp[order], ep[order], Tp[order], Yp[order]
    for i, xi in enumerate(xtr):
        j = int(np.searchsorted(xs, xi))
        lo, hi = max(0, j - nq // 2), min(len(xs), j + nq // 2)
        e_true[i] = es[lo:hi].mean()
        for arm in (0, 1):
            m = Ts[lo:hi] == arm
            if m.sum() < 5: continue
            yy = Ys[lo:hi][m]
            qa = float(np.quantile(yy, a_plus)); q_true[i, arm] = qa
            mlo_true[i, arm] = float(yy[yy <= qa].sum() / len(yy))
            mhi_true[i, arm] = float(yy[yy >= qa].sum() / len(yy))

    def rmse(a_, b_): return float(np.sqrt(np.mean((np.asarray(a_) - np.asarray(b_)) ** 2)))

    print("%s gamma=%.1f Gamma=%.1f alpha+=%.4f | n_train=%d, %d population neighbours"
          % (a.data, a.gamma, G, a_plus, len(xtr), nq), flush=True)
    print("distinct Y values in train: %d  (Y0 is two-valued by construction)"
          % len(np.unique(np.round(Yraw, 6))), flush=True)
    print("%-12s %9s %9s %9s %9s" % ("estimator", "e", "quantile", "mu_lo", "mu_hi"))
    print("-" * 52)
    out = {}
    for tag, eta in (("kNN k=15", H.nuisances_knn(X, T, Y, X, a_plus, k=15)),
                     ("kNN k=50", H.nuisances_knn(X, T, Y, X, a_plus, k=50)),
                     ("neural", H.nuisances_nn(X, T, Y, X, a_plus, seed=a.seed))):
        r = (rmse(eta["e"][:, 1], e_true), rmse(eta["q"], q_true),
             rmse(eta["mu_lo"], mlo_true), rmse(eta["mu_hi"], mhi_true))
        out[tag] = r
        print("%-12s %9.4f %9.4f %9.4f %9.4f" % (tag, *r), flush=True)
    print("\n(lower is better; e is the true propensity stored in the npz, the rest are "
          "population conditional values)")
    json.dump({"data": a.data, "gamma": a.gamma, "Gamma": G, "alpha_plus": a_plus,
               "rmse": out}, open(HERE / ("diag_nuisance_%s_g%g.json" % (a.data, a.gamma)), "w"),
              indent=1)


if __name__ == "__main__":
    main()
