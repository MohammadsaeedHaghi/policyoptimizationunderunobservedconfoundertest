#!/usr/bin/env python3
"""Is the Kallus REGRET OBJECTIVE itself wrong, or is only the optimiser failing?

diag_kallus.py showed `fit_kallus` converging to worst-case regret exactly 0.0 with pi -> 0,
i.e. it returns the never-treat baseline. Regret 0 is the baseline's own score, so the question
is whether any better policy scores BELOW 0 under this objective. If the oracle policy scores
negative and the optimiser still returns 0, the outer loop is at fault. If the oracle policy
scores POSITIVE, the objective is miscalibrated and no optimiser would help.

So: evaluate `_inner_worst_case_regret_w` at FIXED policies -- never, all, oracle (CATE > 0),
the IPW-O-W solution, and softmax approximations of the oracle at a range of temperatures --
and print the regret each attains. Also report each policy's true test value, so objective and
truth can be read side by side.
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
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dgp-seed", type=int, default=0)
    a = ap.parse_args()

    npz = str(HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, a.gamma, a.dgp_seed)))
    R._init(npz)
    G = float(np.exp(2 * a.gamma))
    d = R.draw(a.seed)
    X, T, Yraw, Xte = d["Xtr"], d["Ttr"], d["Ytr"], d["Xte"]
    Y0t, Y1t, cate = d["Y0te"], d["Y1te"], d["cate_te"]
    ym, ys = float(np.mean(Yraw)), float(np.std(Yraw)) or 1.0
    Y = (Yraw - ym) / ys
    common = R._W["common"]
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
    wn, _ = common.ipw_weights_from_data(X, T, 2)

    orc = (cate > 0).astype(float)
    ref = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
           "never": float(np.mean(Y0t)), "all": float(np.mean(Y1t))}
    ref["bc"] = max(ref["never"], ref["all"]); ref["sc"] = ref["oracle"] - ref["bc"]

    # the oracle decision boundary in TRAINING space: sign of the CATE as a function of the index
    Ptr = R._W["pop"]
    xtr = X.ravel()
    tau_tr = np.interp(xtr, np.sort(Xte), (cate)[np.argsort(Xte)])   # CATE along the index
    orc_tr = (tau_tr > 0).astype(float)

    import importlib.util
    kal_mod = sys.modules["fit_kallus"]
    inner = kal_mod._inner_worst_case_regret_w

    def regret_of(pi1_tr):
        """Worst-case regret of a fixed policy given by its P(treat|x) on the TRAINING rows."""
        pi1 = np.clip(np.asarray(pi1_tr, float).ravel(), 0.0, 1.0)
        pi_t = np.where(T == 1, pi1, 1.0 - pi1)          # probability of the OBSERVED arm
        _, reg = inner(X, T, Y, pi_t, wraw, n_arms=2, Gamma=G, wasserstein=False)
        return float(reg)

    def value_of(pi1_te):
        p = np.clip(np.asarray(pi1_te, float).ravel(), 0.0, 1.0)
        return float(np.mean(p * Y1t + (1 - p) * Y0t))

    print("%s gamma=%.1f Gamma=%.4f | oracle %.4f never %.4f all %.4f headroom %.4f"
          % (a.data, a.gamma, G, ref["oracle"], ref["never"], ref["all"], ref["sc"]), flush=True)
    print("%-22s %12s %12s %10s" % ("fixed policy", "regret", "test value", "norm"))
    print("-" * 60)

    rows = []

    def show(tag, pi_tr, pi_te):
        rg, v = regret_of(pi_tr), value_of(pi_te)
        nz = (v - ref["bc"]) / ref["sc"]
        rows.append({"tag": tag, "regret": rg, "value": v, "norm": nz})
        print("%-22s %12.5f %12.4f %+10.3f" % (tag, rg, v, nz), flush=True)

    show("never treat", np.zeros(len(T)), np.zeros(len(Y1t)))
    show("all treat", np.ones(len(T)), np.ones(len(Y1t)))
    show("oracle 1{CATE>0}", orc_tr, orc)
    # softmax approximations of the oracle boundary at several temperatures
    for temp in (0.05, 0.2, 1.0):
        s_tr = 1.0 / (1.0 + np.exp(-tau_tr / temp))
        s_te = 1.0 / (1.0 + np.exp(-cate / temp))
        show("softmax(CATE/%.2f)" % temp, s_tr, s_te)

    try:
        eps = tuple(common.tight_epsilon(common.pairwise_distance_matrix(X), T, wn, 2,
                                         is_distance=True, c_eps=1.0))
        r = R._W["S"]["IPW-O-W"](X, T, Y, wn, n_arms=2, Gamma=G, discretize=False,
                                 lipschitz=None, zscore=False, epsilon=eps)
        pe = R._shapley_fast(Xte.reshape(-1, 1), *R._W["extract"](X, r.pi[1]))
        show("IPW-O-W solution", r.pi[1], pe)
    except Exception as ex:
        print("IPW-O-W failed: %s" % str(ex)[:140], flush=True)

    json.dump({"data": a.data, "gamma": a.gamma, "Gamma": G, "refs": ref, "rows": rows},
              open(HERE / ("diag_kallus2_g%g.json" % a.gamma), "w"), indent=1)
    best = min(rows, key=lambda r_: r_["regret"])
    print("\nlowest regret: %s (regret %.5f, norm %+.3f)" % (best["tag"], best["regret"], best["norm"]))
    print("=> if 'never treat' has the lowest regret while another policy has a much higher\n"
          "   normalised value, the OBJECTIVE prefers the baseline and the optimiser is right.")


if __name__ == "__main__":
    main()
