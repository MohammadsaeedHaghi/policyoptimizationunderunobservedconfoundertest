#!/usr/bin/env python3
"""Is the Kallus sign convention right? Kallus & Zhou's native Y is a RISK (lower is better);
ours is a REWARD (higher is better), and `fit_kallus` switches by negating Y internally
(`maximize=False` => Yc = -Y). A silent mismatch would make the baseline learn the anti-policy,
which is exactly the shape of "Kallus looks bad everywhere".

Four checks, in increasing strength:

  1 INVARIANCE   fit(Y, maximize=True) must give the SAME theta as fit(-Y, maximize=False).
                 These are two spellings of one problem; any difference is a bug.
  2 DIRECTION    corr(learned treat-probability, true CATE) must be POSITIVE under maximize=True.
                 Negative means the method is systematically treating the wrong units.
  3 DELIBERATE FLIP
                 fit(-Y, maximize=True) is the WRONG convention on purpose. It must score far
                 worse than the right one. If flipping barely changes the answer, the fit is not
                 responding to the outcome at all and neither number means anything.
  4 BASELINE     the regret is defined against arm 0 ("treat nobody"), hard-coded via 1[T_i=0].
                 Re-run with the arms RELABELLED so the baseline becomes "treat everybody". If
                 Kallus floors at whichever policy is called the baseline, then "Kallus = best
                 constant" is partly an artefact of that choice, not of the outcome orientation.

Reports each policy's true test value and normalised value, so the tables can be read directly.
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
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)
    common = R._W["common"]
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)

    orc = (cate > 0).astype(float)
    bc = max(float(np.mean(Y0t)), float(np.mean(Y1t)))
    sc = float(np.mean(orc * Y1t + (1 - orc) * Y0t)) - bc
    fit, pred = R._W["kallus"], R._W["kpred"]
    Xt = Xte.reshape(-1, 1)

    def ev(pe):
        v = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
        return v, (v - bc) / sc, float(np.corrcoef(pe, cate)[0, 1]) if pe.std() > 1e-12 else float("nan")

    print("%s gamma=%.1f Gamma=%.3f | oracle %.4f never %.4f all %.4f headroom %.4f"
          % (a.data, a.gamma, G, bc + sc, float(np.mean(Y0t)), float(np.mean(Y1t)), sc), flush=True)
    print("%-34s %9s %8s %9s %9s" % ("variant", "value", "norm", "corrCATE", "regret"))
    print("-" * 74)
    out = {}

    def show(tag, pe, reg):
        v, nz, cc = ev(pe)
        out[tag] = {"value": v, "norm": nz, "corr_cate": cc, "regret": reg,
                    "pe_mean": float(pe.mean())}
        print("%-34s %9.4f %+8.3f %+9.3f %9.5f" % (tag, v, nz, cc, reg), flush=True)

    # 1 + 2 : the shipped convention
    rA = fit(X, T, Y, wraw, n_arms=2, Gamma=G, maximize=True, seed=a.seed)
    peA = pred(rA.theta, Xt)[:, 1]
    show("Y, maximize=True  (as shipped)", peA, float(rA.objective_value))

    # 1 : the same problem spelled the other way
    rB = fit(X, T, -Y, wraw, n_arms=2, Gamma=G, maximize=False, seed=a.seed)
    peB = pred(rB.theta, Xt)[:, 1]
    show("-Y, maximize=False (equivalent)", peB, float(rB.objective_value))
    dth = float(np.abs(np.asarray(rA.theta) - np.asarray(rB.theta)).max())
    print("  => INVARIANCE  max|theta_A - theta_B| = %.3e  %s"
          % (dth, "OK" if dth < 1e-8 else "*** MISMATCH ***"), flush=True)

    # 3 : deliberately wrong convention
    rC = fit(X, T, -Y, wraw, n_arms=2, Gamma=G, maximize=True, seed=a.seed)
    peC = pred(rC.theta, Xt)[:, 1]
    show("-Y, maximize=True  (WRONG on purpose)", peC, float(rC.objective_value))

    # 4 : relabel the arms so the regret baseline becomes "treat everybody"
    Tf = 1 - np.asarray(T).astype(int)
    wf, _ = common.ipw_weights_from_data(X, Tf, 2, normalize=False)
    rD = fit(X, Tf, Y, wf, n_arms=2, Gamma=G, maximize=True, seed=a.seed)
    peD = 1.0 - pred(rD.theta, Xt)[:, 1]      # arm 1 in relabelled space == untreated
    show("arms swapped (baseline=all-treat)", peD, float(rD.objective_value))

    json.dump({"data": a.data, "gamma": a.gamma, "Gamma": G, "seed": a.seed,
               "never": float(np.mean(Y0t)), "all": float(np.mean(Y1t)),
               "oracle": bc + sc, "theta_gap": dth, "rows": out},
              open(HERE / ("diag_kallus5_%s_g%g.json" % (a.data, a.gamma)), "w"), indent=1)


if __name__ == "__main__":
    main()
