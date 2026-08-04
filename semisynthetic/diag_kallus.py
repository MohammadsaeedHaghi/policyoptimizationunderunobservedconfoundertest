#!/usr/bin/env python3
"""Why is Kallus at or below the best constant policy on every cell -- including gamma=0?

At gamma=0 the MSM box collapses to {w_hat}, so `fit_kallus` degenerates to ordinary IPW policy
learning over a softmax class. It should recover most of the headroom there; it recovers none.
That points at the OUTER optimisation, not at the robust inner problem. This script separates the
candidate causes on a single cell, changing one thing at a time:

  base      as the campaign ran it: n_iters=12, n_restarts=2, soft evaluation
  hard      identical fit, but the policy THRESHOLDED at 0.5 before evaluation -- the LP methods
            return near-vertex pi, so a smooth softmax is handicapped by the shared evaluator
  iters     n_iters 12 -> 200 (subgradient descent with a 1/sqrt(k) step needs far more than 12)
  restarts  n_restarts 2 -> 8
  eta       larger step (the self-normalised w* makes the gradient O(1/n_arm) small)
  raw Y     Kallus fed the UNCENTRED outcome (centring is required by our LPs, not by Kallus)
  noavg     final iterate instead of the Polyak average (averaging over 12 iterates from a random
            start keeps most of the random init)

Reports for each: mean/sd of the deployed treat-probability, the fraction that is decisive
(<0.05 or >0.95), the soft and hard test value, and both normalised against the same references
the campaign uses. Read-only w.r.t. the campaign; writes nothing but its own JSON.
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semisynth as R


def summarise(tag, pe, d, ref, extra=""):
    Y0t, Y1t = d["Y0te"], d["Y1te"]
    soft = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
    hp = (pe > 0.5).astype(float)
    hard = float(np.mean(hp * Y1t + (1 - hp) * Y0t))
    bc, sc = ref["bc"], ref["sc"]
    return {"tag": tag, "pe_mean": float(pe.mean()), "pe_sd": float(pe.std()),
            "frac_decisive": float(((pe < 0.05) | (pe > 0.95)).mean()),
            "soft": soft, "hard": hard,
            "soft_norm": (soft - bc) / sc, "hard_norm": (hard - bc) / sc, "extra": extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="german_credit")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dgp-seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "diag_kallus.json"))
    a = ap.parse_args()

    npz = str(HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, a.gamma, a.dgp_seed)))
    R._init(npz)
    G = float(np.exp(2 * a.gamma))
    d = R.draw(a.seed)
    X, T, Yraw, Xte = d["Xtr"], d["Ttr"], d["Ytr"], d["Xte"]
    Y0t, Y1t = d["Y0te"], d["Y1te"]
    ym, ys = float(np.mean(Yraw)), float(np.std(Yraw)) or 1.0
    Y = (Yraw - ym) / ys                                   # what the campaign feeds every method
    common = R._W["common"]
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)

    orc = (d["cate_te"] > 0).astype(float)
    ref = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
           "never": float(np.mean(Y0t)), "all": float(np.mean(Y1t))}
    ref["bc"] = max(ref["never"], ref["all"]); ref["sc"] = ref["oracle"] - ref["bc"]
    print("%s gamma=%.1f Gamma=%.3f | oracle %.4f never %.4f all %.4f headroom %.4f"
          % (a.data, a.gamma, G, ref["oracle"], ref["never"], ref["all"], ref["sc"]), flush=True)

    fit, pred = R._W["kallus"], R._W["kpred"]
    rows = []

    def run(tag, **kw):
        yy = kw.pop("_Y", Y)
        t = time.time()
        r = fit(X, T, yy, wraw, n_arms=2, Gamma=G, maximize=True, seed=a.seed, **kw)
        pe = pred(r.theta, Xte.reshape(-1, 1))[:, 1]
        s = summarise(tag, pe, d, ref, extra="regret=%.4f |theta|=%.3f %.0fs"
                      % (r.objective_value, float(np.abs(r.theta).max()), time.time() - t))
        rows.append(s)
        print("  %-10s pe %.3f+-%.3f dec %.2f | soft %.4f (%+.3f) hard %.4f (%+.3f) | %s"
              % (tag, s["pe_mean"], s["pe_sd"], s["frac_decisive"], s["soft"], s["soft_norm"],
                 s["hard"], s["hard_norm"], s["extra"]), flush=True)

    run("base")                                            # exactly the campaign settings
    run("iters200", n_iters=200)
    run("restart8", n_iters=200, n_restarts=8)
    run("eta10", n_iters=200, eta0=10.0)
    run("eta100", n_iters=200, eta0=100.0)
    run("rawY", n_iters=200, _Y=Yraw)
    run("noavg_hack", n_iters=1)                           # n_iters=1 -> average == the one iterate

    # reference points from the same split, for scale
    S = R._W["S"]
    try:
        wn, _ = common.ipw_weights_from_data(X, T, 2)
        Dm = common.pairwise_distance_matrix(X)
        eps = tuple(common.tight_epsilon(Dm, T, wn, 2, is_distance=True, c_eps=1.0))
        v = S["IPW-O-W"](X, T, Y, wn, n_arms=2, Gamma=G, discretize=False, lipschitz=None,
                         zscore=False, epsilon=eps)
        pe = R._shapley_fast(Xte.reshape(-1, 1), *R._W["extract"](X, v))
        s = summarise("IPW-O-W", np.asarray(pe, float), d, ref); rows.append(s)
        print("  %-10s pe %.3f+-%.3f dec %.2f | soft %.4f (%+.3f) hard %.4f (%+.3f)"
              % ("IPW-O-W", s["pe_mean"], s["pe_sd"], s["frac_decisive"], s["soft"],
                 s["soft_norm"], s["hard"], s["hard_norm"]), flush=True)
    except Exception as ex:
        print("  IPW-O-W reference failed: %s" % str(ex)[:120], flush=True)

    json.dump({"data": a.data, "gamma": a.gamma, "Gamma": G, "seed": a.seed,
               "dgp_seed": a.dgp_seed, "refs": ref, "rows": rows}, open(a.out, "w"), indent=1)
    print("wrote %s" % a.out, flush=True)


if __name__ == "__main__":
    main()
