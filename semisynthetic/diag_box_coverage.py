#!/usr/bin/env python3
"""Does the MSM box actually cover the TRUE inverse weight -- and does it matter that the campaign
passes Hajek-normalised weights rather than raw ones?

Setup. The true propensity is e(x,u) = P(T=1 | x,u) and depends on the hidden state, so the true
inverse weight is w*_i = 1 / P(T=t_i | x_i, u_i). The solvers never see it. They get a NOMINAL
weight w_hat_i = 1 / Phat(T=t_i | x_i) -- conditioning on x ONLY -- and the MSM box is supposed to
be a set around w_hat that contains w*.

The concern. `marginal_sensitivity_box` builds the Tan/Rosenbaum interval
    {1 + t (w_hat - 1) : t in [1/Gamma, Gamma]},
which is derived for the RAW weight w_hat = 1/Phat >= 1, because (w_hat - 1) is the ODDS of not
receiving the observed arm. Under a per-arm Hajek rescaling w_hat -> c_k * w_hat, the quantity
(c_k*w_hat - 1) is NOT c_k*(w_hat - 1), so the box is not the rescaled box -- it is a different set.
`run_semisynth.py` passes the Hajek weights to IPW-O-X / IPW-O-W and the raw ones to Hajek-* and
Kallus, so this is live for the headline methods.

What is measured, at the matched Gamma, per cell:
  c_k               the per-arm Hajek scale factor (approx 1 if raw weights already sum to n)
  frac(w_hat < 1)   Hajek weights below 1, where the Tan endpoints swap
  coverage          fraction of units whose TRUE w* lies inside the box, raw vs Hajek
  width             mean box width, raw vs Hajek
Coverage is the point: a box that does not contain the truth is not a valid sensitivity region, and
one that is wider than necessary just costs conservatism.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semisynth as R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="bank_marketing")
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dgp-seed", type=int, default=0)
    a = ap.parse_args()

    npz = str(HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, a.gamma, a.dgp_seed)))
    R._init(npz)
    common = R._W["common"]
    from common.sensitivity import marginal_sensitivity_box
    G = float(np.exp(2 * a.gamma))

    # rebuild the exact training split the runner uses, and recover the TRUE propensity for it
    P = R._W["pop"]
    x_all, e_all = P["x"].ravel(), P["e"].ravel()
    rng = np.random.default_rng(10_000 + a.seed)
    n = len(x_all)
    T_all = (rng.uniform(size=n) < e_all).astype(int)
    perm = rng.permutation(n)
    tr = perm[:R.N_TRAIN]
    X = x_all[tr].reshape(-1, 1); T = T_all[tr]; e_true = e_all[tr]

    # true inverse weight at the OBSERVED arm: 1/e if treated, 1/(1-e) if control
    w_star = np.where(T == 1, 1.0 / e_true, 1.0 / (1.0 - e_true))

    w_raw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
    w_haj, _ = common.ipw_weights_from_data(X, T, 2, normalize=True)

    print("%s gamma=%.1f Gamma=%.3f  n=%d" % (a.data, a.gamma, G, len(T)))
    for k in (0, 1):
        m = T == k
        print("  arm %d: n=%3d  raw sum=%7.2f (target %d)  Hajek scale c_%d = %.4f"
              % (k, m.sum(), w_raw[m].sum(), len(T), k, (w_haj[m] / w_raw[m]).mean()))
    print("  frac(raw   < 1) = %.4f   min %.4f" % ((w_raw < 1).mean(), w_raw.min()))
    print("  frac(Hajek < 1) = %.4f   min %.4f" % ((w_haj < 1).mean(), w_haj.min()))

    print("\n%-22s %10s %10s %12s %12s" % ("box built on", "coverage", "mean width", "mean lower", "mean upper"))
    print("-" * 70)
    out = {}
    for tag, wh in (("raw 1/Phat(t|x)", w_raw), ("Hajek-normalised", w_haj)):
        lo, hi = marginal_sensitivity_box(wh, G)
        # compare like with like: the Hajek box lives on the rescaled scale, so put w* on it too
        scale = np.ones_like(wh)
        if tag.startswith("Hajek"):
            for k in (0, 1):
                m = T == k
                scale[m] = (w_haj[m] / w_raw[m]).mean()
        cov = float((( w_star * scale >= lo - 1e-9) & (w_star * scale <= hi + 1e-9)).mean())
        out[tag] = {"coverage": cov, "width": float(np.mean(hi - lo)),
                    "lo": float(lo.mean()), "hi": float(hi.mean())}
        print("%-22s %10.4f %10.3f %12.3f %12.3f"
              % (tag, cov, np.mean(hi - lo), lo.mean(), hi.mean()))

    # what the correct Hajek-scale box would be: rescale the RAW box, do not rebuild it
    lo_r, hi_r = marginal_sensitivity_box(w_raw, G)
    sc = np.ones_like(w_raw)
    for k in (0, 1):
        m = T == k
        sc[m] = (w_haj[m] / w_raw[m]).mean()
    cov_c = float(((w_star * sc >= lo_r * sc - 1e-9) & (w_star * sc <= hi_r * sc + 1e-9)).mean())
    print("%-22s %10.4f %10.3f %12.3f %12.3f"
          % ("raw box, then scaled", cov_c, np.mean((hi_r - lo_r) * sc),
             np.mean(lo_r * sc), np.mean(hi_r * sc)))
    out["raw box then scaled"] = {"coverage": cov_c}

    # OPERATIVE Gamma: the smallest Gamma whose box would contain every true weight. Solving
    # w* = 1 + t (w_hat - 1) for t gives the per-unit inflation the truth actually demands; the
    # box needs t in [1/Gamma, Gamma], so the requirement is max(max t, 1/min t).
    tt = (w_star - 1.0) / np.maximum(w_raw - 1.0, 1e-12)
    tt = tt[np.isfinite(tt) & (tt > 0)]
    g_op = float(max(tt.max(), 1.0 / tt.min()))
    print("\n  declared Gamma = %.3f   OPERATIVE Gamma (covers every unit) = %.3f   ratio %.2fx"
          % (G, g_op, g_op / G))
    for q in (0.99, 0.995, 1.0):
        need = float(max(np.quantile(tt, q), 1.0 / np.quantile(tt, 1 - q)))
        print("    Gamma needed to cover %5.1f%% of units: %8.3f" % (100 * q, need))
    out["operative_Gamma"] = g_op
    json.dump({"data": a.data, "gamma": a.gamma, "Gamma": G, "rows": out},
              open(HERE / ("diag_box_%s_g%g.json" % (a.data, a.gamma)), "w"), indent=1)


if __name__ == "__main__":
    main()
