#!/usr/bin/env python3
"""How much does the Kallus OUTER OPTIMISER leave on the table?

For 2 arms with the affine basis on a scalar x, the softmax policy collapses to
    pi_1(x) = sigmoid(s*x + b),
so the whole policy class is TWO parameters. diag_kallus2 showed the regret objective is
correctly calibrated at Gamma=1 (the oracle boundary scores negative regret, never-treat scores
0) yet `fit_kallus` returns 0. A two-parameter problem is small enough to settle by brute force:
grid (s, b), evaluate the exact same `_inner_worst_case_regret_w` objective, and report

    best-on-grid   the lowest regret in the class, and the true test value there
    fit_kallus     what the shipped optimiser actually returns

The gap between them is the implementation defect, isolated from the objective and from the
policy class. Run across datasets and gamma so the campaign-wide impact can be quantified.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semisynth as R

SGRID = np.concatenate([-np.geomspace(40, 0.05, 22), [0.0], np.geomspace(0.05, 40, 22)])
BGRID = np.linspace(-6.0, 6.0, 41)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="german_credit")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dgp-seed", type=int, default=0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    npz = str(HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, a.gamma, a.dgp_seed)))
    R._init(npz)
    G = float(np.exp(2 * a.gamma))
    d = R.draw(a.seed)
    X, T, Yraw, Xte = d["Xtr"], d["Ttr"], d["Ytr"], d["Xte"]
    Y0t, Y1t, cate = d["Y0te"], d["Y1te"], d["cate_te"]
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)
    common = R._W["common"]
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)

    orc = (cate > 0).astype(float)
    bc = max(float(np.mean(Y0t)), float(np.mean(Y1t)))
    sc = float(np.mean(orc * Y1t + (1 - orc) * Y0t)) - bc
    inner = sys.modules["fit_kallus"]._inner_worst_case_regret_w
    xtr, xte = X.ravel(), np.asarray(Xte, float).ravel()

    def regret(s, b):
        p1 = 1.0 / (1.0 + np.exp(-np.clip(s * xtr + b, -40, 40)))
        pi_t = np.where(T == 1, p1, 1.0 - p1)
        _, rg = inner(X, T, Y, pi_t, wraw, n_arms=2, Gamma=G, wasserstein=False)
        return float(rg)

    def value(s, b):
        p = 1.0 / (1.0 + np.exp(-np.clip(s * xte + b, -40, 40)))
        return float(np.mean(p * Y1t + (1 - p) * Y0t))

    t0 = time.time()
    best = (np.inf, None, None)
    for s in SGRID:
        for b in BGRID:
            rg = regret(s, b)
            if rg < best[0]:
                best = (rg, float(s), float(b))
    grid_s = time.time() - t0
    br, bs, bb = best
    gv = value(bs, bb)

    r = R._W["kallus"](X, T, Y, wraw, n_arms=2, Gamma=G, maximize=True, seed=a.seed)
    pe = R._W["kpred"](r.theta, Xte.reshape(-1, 1))[:, 1]
    fv = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
    # the shipped fit expressed in the same 2 parameters
    th = np.asarray(r.theta, float)
    fs, fb = float(th[1, 0] - th[0, 0]), float(th[1, 1] - th[0, 1])

    out = {"data": a.data, "gamma": a.gamma, "Gamma": G, "seed": a.seed, "dgp_seed": a.dgp_seed,
           "bc": bc, "sc": sc,
           "grid": {"regret": br, "s": bs, "b": bb, "value": gv, "norm": (gv - bc) / sc,
                    "n_evals": int(len(SGRID) * len(BGRID)), "seconds": round(grid_s, 1)},
           "fit": {"regret": float(r.objective_value), "s": fs, "b": fb, "value": fv,
                   "norm": (fv - bc) / sc, "regret_recomputed": regret(fs, fb)}}
    print("%-16s g=%.1f G=%8.3f | GRID regret %9.5f norm %+7.3f (s=%.2f b=%.2f) | "
          "FIT regret %9.5f norm %+7.3f (s=%.2f b=%.2f) | gap %+.3f  [%d evals, %.0fs]"
          % (a.data, a.gamma, G, br, out["grid"]["norm"], bs, bb,
             out["fit"]["regret"], out["fit"]["norm"], fs, fb,
             out["grid"]["norm"] - out["fit"]["norm"], out["grid"]["n_evals"], grid_s), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
