#!/usr/bin/env python3
"""Sharp box-only baseline + high-precision references for the KZ18 campaign (license-free).

Same two-point sharp adversary as assets/exp_gstar/run_sharp.py and run_sharp_kmz.py (MSM box
PLUS the sharpness equality E[W | T=t, cell] = 1/e_hat, solved in closed form by sorting),
binned at BINS=15 over the scalar index. Unlike the KMZ campaign there is no analytic bin mean
here (the outcome depends on the full 5-vector X5, not on the scalar), so every policy is
evaluated on a FIXED large test draw with known counterfactuals -- the same evaluation the
paper uses, at 200k units instead of their test set.

Also computes the references the runner cannot provide at infinite data: binned-naive and
oracle values in both regimes (uncapped and capped 30%), for both index reductions.

Outputs: kz_sharp.json   (sharp values, both indices x both regimes, per-seed policies saved)
         kz_refs.json    (oracle / naive / never / all, uncapped and capped, both indices)
"""
import json, sys
from pathlib import Path
import importlib.util
import numpy as np

HERE = Path(__file__).resolve().parent

def _load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m

def sharp_min_arm(Y, w_hat, Gamma):
    """min over the sharp MSM adversary of (1/n) sum_i W_i Y_i, W in [a,b], mean(W) = w_hat."""
    if len(Y) == 0: return 0.0
    a = 1.0 + (w_hat - 1.0) / Gamma
    b = 1.0 + Gamma * (w_hat - 1.0)
    if b - a < 1e-12: return float(w_hat * np.mean(Y))
    lam = (w_hat - a) / (b - a)
    ys = np.sort(np.asarray(Y, float)); n = len(ys); k = lam * n
    kf, kfrac = int(np.floor(k)), k - int(np.floor(k))
    lo = ys[:kf].sum() + (ys[kf] * kfrac if kf < n else 0.0)
    return float((b * lo + a * (ys.sum() - lo)) / n)

def sharp_scores(Y, T, ci, ncell, Gamma):
    m1, m0 = np.full(ncell, np.nan), np.full(ncell, np.nan)
    for j in range(ncell):
        m = ci == j
        if m.sum() == 0: continue
        e = float(np.clip(T[m].mean(), 0.02, 0.98))
        yt, yc = Y[m & (T == 1)], Y[m & (T == 0)]
        m1[j] = sharp_min_arm(yt, 1.0 / e, Gamma) * (len(yt) / max(m.sum(), 1))
        m0[j] = sharp_min_arm(yc, 1.0 / (1 - e), Gamma) * (len(yc) / max(m.sum(), 1))
    return m1, m0

def policies(score, mass, cap=None):
    if cap is None: return (score > 0).astype(float)
    pi = np.zeros(len(score)); rem = cap
    for j in sorted(range(len(score)), key=lambda i: -score[i]):
        if score[j] <= 0 or rem <= 1e-12: break
        take = min(1.0, rem / max(mass[j], 1e-12)); pi[j] = take; rem -= take * mass[j]
    return pi

GAMMAS = [1.0, 2.0, 3.0, 4.4817, 6.0, 8.0]
SEEDS = list(range(5))
BINS = 15
NTRAIN = 200                                      # match the campaign: the paper's own n
NTEST = 200000
ARMS = (("main", "dgp.py"), ("cidx", "dgp_cidx.py"))

out = {"method": "SharpIPW-O-X", "gammas": GAMMAS, "seeds": SEEDS, "bins": BINS,
       "n_train": NTRAIN, "n_test": NTEST, "arms": {}}
refs = {}

for arm, fn in ARMS:
    d = _load(HERE / fn, "kz_sharp_" + arm)
    te, ft = d.generate(NTEST, 999)
    xte = te["X"].ravel(); Y1, Y0 = ft["Y1"], ft["Y0"]
    edges = np.quantile(xte, np.linspace(0, 1, BINS + 1))
    edges[0] -= 1e-9; edges[-1] += 1e-9
    cte = np.clip(np.digitize(xte, edges) - 1, 0, BINS - 1)
    massT = np.array([(cte == j).mean() for j in range(BINS)])
    def val(pi_bins):
        p = np.asarray(pi_bins, float)[cte]
        return float(np.mean(p * Y1 + (1 - p) * Y0))

    out["arms"][arm] = {"gstar": float(d.GSTAR), "index": d.INDEX,
                        "bin_edges": [round(float(v), 5) for v in edges], "regimes": {}}
    for reg, cap in (("uncap", None), ("cap", 0.3)):
        vals = {gi: [] for gi in range(len(GAMMAS))}; polS = {}
        for sd in SEEDS:
            obs, _ = d.generate(NTRAIN, sd)
            X, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
            ci = np.clip(np.digitize(X, edges) - 1, 0, BINS - 1)
            polS[str(sd)] = {}
            for gi, g in enumerate(GAMMAS):
                m1, m0 = sharp_scores(Y, T, ci, BINS, g)
                sc = np.nan_to_num(m1, nan=-1e9) - np.nan_to_num(m0, nan=0.0)
                pi = policies(sc, massT, cap)
                vals[gi].append(val(pi))
                polS[str(sd)]["%g" % g] = [round(float(v), 4) for v in pi]
        out["arms"][arm]["regimes"][reg] = {
            "mean": {"SharpIPW-O-X": [round(float(np.mean(vals[gi])), 4) for gi in range(len(GAMMAS))]},
            "sd": {"SharpIPW-O-X": [round(float(np.std(vals[gi])), 4) for gi in range(len(GAMMAS))]},
            "policy_by_seed": {"SharpIPW-O-X": polS}}

    # ---- infinite-data references on the same evaluation draw ----
    ob, fb = d.generate(NTEST, 12345)                      # independent "infinite" sample
    xb, Tb, Yb = ob["X"].ravel(), ob["T"], ob["Y"]
    cb = np.clip(np.digitize(xb, edges) - 1, 0, BINS - 1)
    hat = np.zeros(BINS)
    for j in range(BINS):
        m = cb == j; yt, yc = Yb[m & (Tb == 1)], Yb[m & (Tb == 0)]
        if len(yt) > 5 and len(yc) > 5: hat[j] = yt.mean() - yc.mean()
    cate_b = np.array([d.cate(0.5 * (edges[j] + edges[j + 1])) for j in range(BINS)])
    refs[arm] = {
        "index": d.INDEX, "gstar": float(d.GSTAR),
        "oracle_uncap": val(policies(cate_b, massT)), "naive_uncap": val(policies(hat, massT)),
        "oracle_cap30": val(policies(cate_b, massT, 0.3)), "naive_cap30": val(policies(hat, massT, 0.3)),
        "never": float(np.mean(Y0)), "all": float(np.mean(Y1)),
        "oracle_full": float(np.mean(np.maximum(Y1, Y0))),
        "naive_treat_frac": float((policies(hat, massT) * massT).sum()),
        "oracle_treat_frac": float((policies(cate_b, massT) * massT).sum()),
        "corr_xS": float(np.corrcoef(xb, fb["S"])[0, 1]),
        "cate_bins": [round(float(v), 4) for v in cate_b],
        "naive_hat_bins": [round(float(v), 4) for v in hat]}

(HERE / "kz_sharp.json").write_text(json.dumps(out, indent=1))
(HERE / "kz_refs.json").write_text(json.dumps(refs, indent=1))
gi = GAMMAS.index(4.4817)
for arm, _ in ARMS:
    r = refs[arm]
    print("[%s] sharp @ matched: uncap %.3f  cap %.3f" %
          (arm, out["arms"][arm]["regimes"]["uncap"]["mean"]["SharpIPW-O-X"][gi],
           out["arms"][arm]["regimes"]["cap"]["mean"]["SharpIPW-O-X"][gi]))
    print("     oracle %.3f naive %.3f never %.3f all %.3f | capped oracle %.3f naive %.3f | corr(x,S) %.3f"
          % (r["oracle_uncap"], r["naive_uncap"], r["never"], r["all"],
             r["oracle_cap30"], r["naive_cap30"], r["corr_xS"]))
print("SHARP_KZ_DONE")
