#!/usr/bin/env python3
"""Sharp box-only baseline + analytic references for the KMZ'19 DGP campaign (license-free).

Same two-point sharp adversary as assets/exp_gstar/run_sharp.py (box + sharpness equality,
solved by sorting), binned at BINS=15 over x in [-1,1]. Exact evaluation from the analytic
bin means of E[Y(a)|x]: the U-term -2(2U-1)(1+0.5X) averages out under U ~ Bern(1/2) (U is
independent of x in this DGP), so
    E[Y(1)|x] =  X + 1 - 2 sin(2X),   E[Y(0)|x] = -X - 1 + 2 sin(2X),   X = 2x.

Also computes the analytic (infinite-data) references the runner cannot provide:
capped-oracle and capped-naive (top-30%-mass by true CATE / by the naive CATE-hat under the
extremal propensity), plus the uncapped analytic naive.

Outputs: kmz_sharp.json (uncap + cap regimes, Gamma grid incl. the paper's matched values),
         kmz_refs.json (analytic reference values).
"""
import json, sys
from pathlib import Path
import importlib.util
import numpy as np

HERE = Path(__file__).resolve().parent
def _load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m
sys.path.insert(0, str(HERE.parent.parent))
gs = _load(HERE.parent / "exp_gstar" / "run_sharp.py", "gstar_sharp_lib") if False else None
# (self-contained: replicate the two helpers rather than importing the gstar script, which runs on import)

def sharp_min_arm(Y, w_hat, Gamma):
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
        e = T[m].mean(); e = min(max(e, 0.02), 0.98)
        yt, yc = Y[m & (T == 1)], Y[m & (T == 0)]
        m1[j] = sharp_min_arm(yt, 1.0 / e, Gamma) * (len(yt) / max(m.sum(), 1))
        m0[j] = sharp_min_arm(yc, 1.0 / (1 - e), Gamma) * (len(yc) / max(m.sum(), 1))
    return m1, m0

def policies(m1, m0, mass, cap=None):
    score = np.nan_to_num(m1, nan=-1e9) - np.nan_to_num(m0, nan=0.0)
    if cap is None: return (score > 0).astype(float)
    pi = np.zeros(len(score)); rem = cap
    for j in sorted(range(len(score)), key=lambda i: -score[i]):
        if score[j] <= 0 or rem <= 1e-12: break
        take = min(1.0, rem / mass[j]); pi[j] = take; rem -= take * mass[j]
    return pi

d15 = _load(HERE / "dgp_g15.py", "kmz_sharp_g15")
GAMMAS = [1.0, 2.0, 3.0, 4.4817, 6.0, 8.0]
SEEDS = list(range(5))
BINS = 15
edges = np.linspace(-1, 1, BINS + 1)

xf = np.linspace(-1, 1, 40001); Xf = 2 * xf
eY1f = Xf + 1 - 2 * np.sin(2 * Xf)
eY0f = -Xf - 1 + 2 * np.sin(2 * Xf)
bif = np.clip(np.digitize(xf, edges) - 1, 0, BINS - 1)
E1b = np.array([eY1f[bif == j].mean() for j in range(BINS)])
E0b = np.array([eY0f[bif == j].mean() for j in range(BINS)])
def exact_cont(pi_bins): return float(np.mean(pi_bins * E1b + (1 - pi_bins) * E0b))

out = {"method": "SharpIPW-O-X", "gammas": GAMMAS, "bins": edges.tolist(), "seeds": SEEDS,
       "gstar": float(d15.GSTAR), "regimes": {}}
for reg, cap in (("uncap", None), ("cap", 0.3)):
    vals = {gi: [] for gi in range(len(GAMMAS))}; polS = {}
    for sd in SEEDS:
        obs, _ = d15.generate(400, sd)
        X, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
        ci = np.clip(np.digitize(X, edges) - 1, 0, BINS - 1)
        mass = np.array([(ci == j).mean() for j in range(BINS)])
        polS[str(sd)] = {}
        for gi, g in enumerate(GAMMAS):
            m1, m0 = sharp_scores(Y, T, ci, BINS, g)
            pi = policies(m1, m0, mass, cap)
            vals[gi].append(exact_cont(pi))
            polS[str(sd)]["%g" % g] = [round(float(v), 4) for v in pi]
    out["regimes"][reg] = {
        "mean": {"SharpIPW-O-X": [round(float(np.mean(vals[gi])), 4) for gi in range(len(GAMMAS))]},
        "sd": {"SharpIPW-O-X": [round(float(np.std(vals[gi])), 4) for gi in range(len(GAMMAS))]},
        "policy_by_seed": {"SharpIPW-O-X": polS}}
(HERE / "kmz_sharp.json").write_text(json.dumps(out, indent=1))

# sharp at the paper's other Gamma* strengths (matched Gamma only, uncapped)
strength = {}
for tag in ("05", "10"):
    dg = _load(HERE / f"dgp_g{tag}.py", f"kmz_sharp_g{tag}")
    g = float(dg.GSTAR); vv = []
    for sd in SEEDS:
        obs, _ = dg.generate(400, sd)
        X, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
        ci = np.clip(np.digitize(X, edges) - 1, 0, BINS - 1)
        m1, m0 = sharp_scores(Y, T, ci, BINS, g)
        vv.append(exact_cont(policies(m1, m0, np.ones(BINS) / BINS)))
    strength["g" + tag] = {"gstar": g, "mean": round(float(np.mean(vv)), 4), "sd": round(float(np.std(vv)), 4)}
out["strength_uncap"] = strength
(HERE / "kmz_sharp.json").write_text(json.dumps(out, indent=1))

# ---- analytic references ----
cate_f = eY1f - eY0f
po = (cate_f >= max(float(np.quantile(cate_f, 0.70)), 0.0)).astype(float)
# naive-hat under the extremal propensity at each Gamma* (numeric, infinite data)
refs = {"oracle_uncap": float(np.mean(np.maximum(eY1f, eY0f) * 0 + (cate_f > 0) * eY1f + (cate_f <= 0) * eY0f)),
        "oracle_cap30": float(np.mean(po * eY1f + (1 - po) * eY0f)),
        "never": float(np.mean(eY0f)), "all": float(np.mean(eY1f)), "by_gstar": {}}
for tag, dg in (("05", None), ("10", None), ("15", None)):
    dgm = _load(HERE / f"dgp_g{tag}.py", f"kmz_ref_g{tag}")
    e1 = dgm.propensity(xf, np.ones_like(xf)); e0 = dgm.propensity(xf, -np.ones_like(xf))
    pt = 0.5 * e1 + 0.5 * e0
    pU1_T1 = 0.5 * e1 / pt; pU1_T0 = 0.5 * (1 - e1) / (1 - pt)
    # E[Y | T=t, x]: outcome U-term is -2(2U-1)(1+0.5X)
    uterm_T1 = -2 * (2 * pU1_T1 - 1) * (1 + 0.5 * Xf)
    uterm_T0 = -2 * (2 * pU1_T0 - 1) * (1 + 0.5 * Xf)
    hat = (eY1f + uterm_T1) - (eY0f + uterm_T0)
    pnu = (hat > 0).astype(float)
    pnc = (hat >= float(np.quantile(hat, 0.70))).astype(float)
    refs["by_gstar"]["g" + tag] = {
        "gstar": float(dgm.GSTAR),
        "naive_uncap": float(np.mean(pnu * eY1f + (1 - pnu) * eY0f)),
        "naive_cap30": float(np.mean(pnc * eY1f + (1 - pnc) * eY0f))}
(HERE / "kmz_refs.json").write_text(json.dumps(refs, indent=1))
g5i = GAMMAS.index(4.4817)
print("sharp @ matched (g15): uncap %.3f cap %.3f" %
      (out["regimes"]["uncap"]["mean"]["SharpIPW-O-X"][g5i], out["regimes"]["cap"]["mean"]["SharpIPW-O-X"][g5i]))
print("strength:", strength)
print("refs:", json.dumps(refs["by_gstar"], indent=1), "\noracle_cap30 %.3f oracle_uncap %.3f" %
      (refs["oracle_cap30"], refs["oracle_uncap"]))
