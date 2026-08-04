#!/usr/bin/env python3
"""Is SharpHess mis-optimised the same way Kallus was?

SharpHess scores 0.010 normalised at gamma=0 while every other method scores 0.75-0.88 — the same
signature that turned out to be an optimiser defect in Kallus. SharpHess is easier to settle,
because Eq. 15 is LINEAR in pi: over all measurable policies the optimum is the pointwise rule
`treat iff g_i(1) > g_i(0)`, which `sharp_hess_policy` already returns in closed form. So the
paper's own objective has a known optimum to compare against.

Ladder, weakest to strongest claim about where any loss occurs:

  argmax-exact   pointwise argmax of the cross-fitted scores, deployed like every other free-pi
                 method (Shapley/kNN). The optimum of the paper's objective over ALL policies.
  grid-logistic  brute force over the campaign's actual class, pi = sigmoid(s*x + b): two
                 parameters, so a grid settles it exactly. Separates "class too small" from
                 "optimiser failing".
  fit-campaign   learn_policy_parametric(n_iter=300, lr=0.05, restarts=3) — what the campaign ran.
  fit-defaults   learn_policy_parametric(n_iter=800, lr=0.2, restarts=6) — the module defaults,
                 which the campaign did NOT use.
  fit-mlp        hidden=8, closer to the paper's neural policy class.

Also runs the module's own `test_gamma1_is_aipw` identity, which validates the SCORE formulas
independently of any policy optimisation.
"""
import sys, json, argparse
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_semisynth as R

SGRID = np.concatenate([-np.geomspace(60, 0.05, 24), [0.0], np.geomspace(0.05, 60, 24)])
BGRID = np.linspace(-8.0, 8.0, 49)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="german_credit")
    ap.add_argument("--gamma", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dgp-seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    npz = str(HERE / "prepared" / ("%s_g%g_d%d_pop.npz" % (a.data, a.gamma, a.dgp_seed)))
    R._init(npz)
    G = float(np.exp(2 * a.gamma))
    d = R.draw(a.seed)
    X, T, Yraw, Xte = d["Xtr"], d["Ttr"], d["Ytr"], d["Xte"]
    Y0t, Y1t, cate = d["Y0te"], d["Y1te"], d["cate_te"]
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)
    H = R._W["hess"]

    orc = (cate > 0).astype(float)
    bc = max(float(np.mean(Y0t)), float(np.mean(Y1t)))
    sc = float(np.mean(orc * Y1t + (1 - orc) * Y0t)) - bc
    Xt = Xte.reshape(-1, 1); xtr, xte = X.ravel(), np.asarray(Xte, float).ravel()

    g = H.fit_scores(X, T, Y, Gamma=G, k=a.k, n_folds=2, maximize=True, seed=a.seed)
    gap = np.asarray(g)[:, 1] - np.asarray(g)[:, 0]
    base = float(np.mean(np.asarray(g)[:, 0]))

    def obj(p_tr):            # the paper's Eq.-15 objective on the training rows (maximise)
        return float(np.mean(gap * np.asarray(p_tr).ravel())) + base

    def ev(p_te):
        p = np.clip(np.asarray(p_te, float).ravel(), 0, 1)
        v = float(np.mean(p * Y1t + (1 - p) * Y0t))
        return v, (v - bc) / sc

    print("k=%d" % a.k, end=" | ", flush=True)
    print("%s gamma=%.1f Gamma=%.3f | oracle %.4f never %.4f all %.4f headroom %.4f | gap sd %.4f"
          % (a.data, a.gamma, G, bc + sc, float(np.mean(Y0t)), float(np.mean(Y1t)), sc,
             float(np.std(gap))), flush=True)
    print("%-26s %11s %10s %8s" % ("variant", "Eq15 obj", "value", "norm"))
    print("-" * 58)
    rows = {}

    def show(tag, p_tr, p_te):
        o = obj(p_tr); v, nz = ev(p_te)
        rows[tag] = {"obj": o, "value": v, "norm": nz}
        print("%-26s %11.5f %10.4f %+8.3f" % (tag, o, v, nz), flush=True)

    # exact optimum of the paper's objective over ALL policies
    pi_ex = (gap > 0).astype(float)
    pe_ex = R._shapley_fast(Xt, *R._W["extract"](X, pi_ex))
    show("argmax-exact", pi_ex, pe_ex)

    # exact optimum WITHIN the campaign's logistic class
    best = (-np.inf, 0.0, 0.0)
    for s in SGRID:
        for b in BGRID:
            p = 1.0 / (1.0 + np.exp(-np.clip(s * xtr + b, -40, 40)))
            o = float(np.mean(gap * p))
            if o > best[0]:
                best = (o, float(s), float(b))
    _, bs, bb = best
    show("grid-logistic", 1.0 / (1.0 + np.exp(-np.clip(bs * xtr + bb, -40, 40))),
         1.0 / (1.0 + np.exp(-np.clip(bs * xte + bb, -40, 40))))

    for tag, kw in (("fit-FIXED-logistic", dict()),
                    ("fit-FIXED-mlp(h=8)", dict(hidden=8))):
        th = H.learn_policy_parametric(X, g, seed=a.seed, maximize=True, **kw)
        show(tag, H.apply_policy(th, X.reshape(-1, 1)), H.apply_policy(th, Xt))

    print("  grid-logistic best (s, b) = (%.2f, %.2f)" % (bs, bb), flush=True)
    ok = H.test_gamma1_is_aipw()
    print("  test_gamma1_is_aipw (score formulas): %s" % ("PASS" if ok else "*** FAIL ***"), flush=True)

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"data": a.data, "gamma": a.gamma, "Gamma": G, "seed": a.seed,
                   "bc": bc, "sc": sc, "gamma1_aipw_ok": bool(ok),
                   "grid_s": bs, "grid_b": bb, "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
