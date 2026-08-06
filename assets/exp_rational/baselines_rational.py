#!/usr/bin/env python3
"""Kallus and SharpHess on the rational-DM DGP, with the 2026-08-04 fixes.

The confirmation run scored only the O-X / O-W / X-X families and naive. This adds the two
published baselines on the SAME (cfg, seed) draws so the comparison is paired, and it computes
only those two -- the LPs are the expensive part and re-solving them would buy nothing.

Both baselines carry the corrections from that day:
  Kallus     normalised gradient step, restart scales spread over two orders of magnitude, and
             best-ITERATE selection. The old defaults returned a near-uniform softmax.
  SharpHess  the same class of fix in learn_policy_parametric, plus the paper's Table-5 NEURAL
             nuisances ({64,64,32} ReLU, Adam 1e-3, 300 epochs, batch 64, patience 10).

BOTH Hess nuisance arms are run, and there is a real prediction to test. On the semi-synthetic
campaign the neural nuisances HURT under confounding, and the diagnosis was that Y0 there is
two-valued, so the truncated-mean target Y*1{Y <= q(x)} is near-discrete and a smooth regression
averages away what an empirical k-NN quantile reproduces exactly. Here the outcome is CONTINUOUS,
so that mechanism should not fire and the neural arm should do at least as well as k-NN. If it
does not, the semi-campaign diagnosis is wrong.
"""
from __future__ import annotations

import sys, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/home1/haghim/code 1.1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
sys.path.insert(0, str(ROOT / "methods" / "SharpHess"))
sys.path.insert(0, str(HERE))

from dgp_rational import Cfg, draw


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def main():
    ap = argparse.ArgumentParser()
    for k, v in Cfg().as_dict().items():
        ap.add_argument("--" + k, type=type(v), default=v)
    ap.add_argument("--single-seed", type=int, required=True)
    ap.add_argument("--n-train", type=int, default=400)
    ap.add_argument("--n-test", type=int, default=4000)
    ap.add_argument("--tag", default="cfg")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = Cfg(**{k: getattr(a, k) for k in Cfg().as_dict()})

    import common
    import sharp_hess as H
    fit_kallus = _load("methods/Kallus/kallus.py", "fit_kallus")
    predict_kallus = _load("methods/Kallus/kallus.py", "predict_kallus")

    sd = a.single_seed
    tr = draw(a.n_train, 1000 + sd, cfg)          # EXACTLY the draws smoke_rational uses
    te = draw(a.n_test, 90000 + sd, cfg)
    X = tr["x"].reshape(-1, 1); T = tr["T"]; Yraw = tr["Y"]
    Xte, Y0t, Y1t = te["x"], te["Y0"], te["Y1"]
    Y = (Yraw - float(np.mean(Yraw))) / (float(np.std(Yraw)) or 1.0)

    orc = (te["cate_cond"] > 0).astype(float)
    refs = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
            "never": float(np.mean(Y0t)), "all": float(np.mean(Y1t))}
    bc = max(refs["never"], refs["all"]); sc = refs["oracle"] - bc
    G = float(cfg.Gstar)
    Xt = Xte.reshape(-1, 1)

    def ev(pe):
        pe = np.clip(np.asarray(pe, float).ravel(), 0.0, 1.0)
        return float(np.mean(pe * Y1t + (1 - pe) * Y0t))

    out = {"refs": refs, "bc": bc, "sc": sc}
    t0 = time.time()

    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
    try:
        r = fit_kallus(X, T, Y, wraw, n_arms=2, Gamma=G, maximize=True, seed=sd)
        pe = predict_kallus(r.theta, Xt)[:, 1]
        out["Kallus"] = {"value": ev(pe), "objective": float(r.objective_value),
                         "pe_mean": float(pe.mean()), "pe_sd": float(pe.std())}
    except Exception as ex:
        print("FAIL Kallus: %s" % str(ex)[:120], flush=True)

    # SharpHess = PAPER-EXACT (their repo's recipe end to end); the k-NN arm stays as a
    # clearly-labelled diagnostic only.
    try:
        pol = H.hess_paper(X, T, Y, Gamma=G, seed=sd, maximize=True)
        pe = np.clip(H.apply_hess_paper(pol, Xt), 0.0, 1.0)
        out["SharpHess"] = {"value": ev(pe), "pe_mean": float(pe.mean()),
                            "pe_sd": float(pe.std()), "recipe": "paper-exact"}
    except Exception as ex:
        print("FAIL SharpHess(paper): %s" % str(ex)[:120], flush=True)
    try:
        scores = H.fit_scores(X, T, Y, Gamma=G, n_folds=2, maximize=True, seed=sd,
                              nuisance="knn", k=15)
        th = H.learn_policy_parametric(X, scores, seed=sd, maximize=True)
        pe = H.apply_policy(th, Xt)
        out["SharpHess-kNN"] = {"value": ev(pe), "pe_mean": float(pe.mean()),
                                "pe_sd": float(pe.std())}
    except Exception as ex:
        print("FAIL SharpHess-kNN: %s" % str(ex)[:120], flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"tag": a.tag, "cfg": cfg.as_dict(), "seed": sd, "n_train": a.n_train,
               "base": out}, open(a.out, "w"), indent=1)
    print("  %s seed %d: %s  (%.0fs)"
          % (a.tag, sd, {k: round((v["value"] - bc) / sc, 3)
                         for k, v in out.items() if isinstance(v, dict) and "value" in v},
             time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
