#!/usr/bin/env python3
"""Sharp box-only baseline (Sharp-O-X) for the gstar suite -- Dorn-Guo-style SHARP MSM bounds.

The plain box worst case lets the adversary set every unit's weight to a box endpoint
independently. The SHARP version adds the constraint the MSM actually implies,
E[W | T=t, x] = w_hat(x) (weights must average to the fitted inverse propensity within each
(x, t) cell), which pins the mixing fraction: with two-point weights {a, b},
lambda = (w_hat - a) / (b - a) of the arm's mass gets the HIGH weight b, placed adversarially
on the LOWEST outcomes (worst case for that arm's contribution). Closed form by sorting --
no LP, no license.

Policy: per level/bin, sharp-min contribution of treating vs not; uncapped treats where
treat > control; capped fills the budget greedily by the sharp score. Evaluation exact
(discrete grid / analytic bin integrals for the continuous piecewise-constant policy).

Outputs: gstar_sharp.json (discrete, uncap+cap regimes, runner-compatible schema subset)
         gstar_sharp_cont.json (continuous, binned BINS=15, uncapped + capped)
"""
import json, sys
from pathlib import Path
import importlib.util
import numpy as np

HERE = Path(__file__).resolve().parent
def _load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m
d = _load(HERE / "dgp.py", "gstar_sharp_d")
dc = _load(HERE / "dgp_cont.py", "gstar_sharp_dc")

GAMMAS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]
GAMMAS_C = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
SEEDS = list(range(5))
BINS = 15


def sharp_min_arm(Y, w_hat, Gamma):
    """Sharp worst-case (minimum) of mean_i(w_i Y_i) over w in the MSM box with
    mean(w) = w_hat. Two-point solution: b on the lowest lam-fraction, a elsewhere."""
    if len(Y) == 0: return 0.0
    a = 1.0 + (w_hat - 1.0) / Gamma
    b = 1.0 + Gamma * (w_hat - 1.0)
    if b - a < 1e-12: return float(w_hat * np.mean(Y))
    lam = (w_hat - a) / (b - a)
    ys = np.sort(np.asarray(Y, float))
    n = len(ys); k = lam * n
    kf, kfrac = int(np.floor(k)), k - int(np.floor(k))
    lo_sum = ys[:kf].sum() + (ys[kf] * kfrac if kf < n else 0.0)
    hi_sum = ys.sum() - lo_sum
    return float((b * lo_sum + a * hi_sum) / n)


def sharp_scores(Y, T, cell_idx, ncell, Gamma):
    """Per-cell sharp-min treat/control contributions (per-capita within cell)."""
    m1, m0 = np.full(ncell, np.nan), np.full(ncell, np.nan)
    for j in range(ncell):
        m = cell_idx == j
        if m.sum() == 0: continue
        e = T[m].mean()
        e = min(max(e, 1.0 / m.sum()), 1 - 1.0 / m.sum()) if 0 < e < 1 else min(max(e, 0.02), 0.98)
        yt, yc = Y[m & (T == 1)], Y[m & (T == 0)]
        # arm means within the cell: (1/n_j) sum_{T=t} w Y with E[w|T=t]=1/e_t
        m1[j] = sharp_min_arm(yt, 1.0 / e, Gamma) * (len(yt) / max(m.sum(), 1))
        m0[j] = sharp_min_arm(yc, 1.0 / (1 - e), Gamma) * (len(yc) / max(m.sum(), 1))
    return m1, m0


def policies(m1, m0, mass, cap=None):
    """Uncapped: treat where sharp-treat > sharp-control. Capped: greedy by score."""
    score = m1 - m0
    if cap is None:
        return (score > 0).astype(float)
    pi = np.zeros(len(score)); rem = cap
    for j in sorted(range(len(score)), key=lambda i: -(score[i] if score[i] == score[i] else -1e9)):
        if score[j] != score[j] or score[j] <= 0 or rem <= 1e-12: break
        take = min(1.0, rem / mass[j]); pi[j] = take; rem -= take * mass[j]
    return pi


# ---------------- discrete ----------------
LV = np.asarray(d.LEVELS, float); NL = len(LV)
out_d = {"method": "SharpIPW-O-X", "gammas": GAMMAS, "grid": LV.tolist(), "seeds": SEEDS,
         "regimes": {}}
for reg, cap in (("uncap", None), ("cap", 0.3)):
    vals = {gk: [] for gk in range(len(GAMMAS))}
    polS = {}
    for sd in SEEDS:
        obs, _ = d.generate(600, sd)
        X, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
        ci = np.searchsorted(LV, X - 1e-9)
        polS[str(sd)] = {}
        for gi, g in enumerate(GAMMAS):
            m1, m0 = sharp_scores(Y, T, ci, NL, g)
            pi = policies(m1, m0, np.full(NL, 1.0 / NL), cap)
            vals[gi].append(d.exact_value(pi))
            polS[str(sd)]["%g" % g] = [round(float(v), 4) for v in pi]
    out_d["regimes"][reg] = {
        "mean": {"SharpIPW-O-X": [round(float(np.mean(vals[gi])), 4) for gi in range(len(GAMMAS))]},
        "sd": {"SharpIPW-O-X": [round(float(np.std(vals[gi])), 4) for gi in range(len(GAMMAS))]},
        "policy_by_seed": {"SharpIPW-O-X": polS}}
(HERE / "gstar_sharp.json").write_text(json.dumps(out_d, indent=1))
g5 = GAMMAS.index(5.0)
print("discrete Sharp-O-X at Gamma*=5: uncap %.3f  cap %.3f"
      % (out_d["regimes"]["uncap"]["mean"]["SharpIPW-O-X"][g5],
         out_d["regimes"]["cap"]["mean"]["SharpIPW-O-X"][g5]))

# ---------------- continuous (binned) ----------------
edges = np.linspace(-1, 1, BINS + 1); mids = 0.5 * (edges[1:] + edges[:-1])
# analytic per-bin means of eY1/eY0 for exact evaluation of piecewise-constant policies
xf = np.linspace(-1, 1, 40001); sig = lambda z: 1 / (1 + np.exp(-z))
ES = 2 * sig(10 * xf) - 1
eY0f = 8 * ES; eY1f = 9 * ES + 3 * xf - 1
bif = np.clip(np.digitize(xf, edges) - 1, 0, BINS - 1)
E1b = np.array([eY1f[bif == j].mean() for j in range(BINS)])
E0b = np.array([eY0f[bif == j].mean() for j in range(BINS)])
def exact_cont(pi_bins):
    return float(np.mean(pi_bins * E1b + (1 - pi_bins) * E0b))

out_c = {"method": "SharpIPW-O-X", "gammas": GAMMAS_C, "bins": edges.tolist(), "mids": mids.tolist(),
         "seeds": SEEDS, "note": "binned (BINS=%d) sharp box-only; exact analytic evaluation" % BINS,
         "regimes": {}}
for reg, cap in (("uncap", None), ("cap", 0.3)):
    vals = {gi: [] for gi in range(len(GAMMAS_C))}
    polS = {}
    for sd in SEEDS:
        obs, _ = dc.generate(400, sd)
        X, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
        ci = np.clip(np.digitize(X, edges) - 1, 0, BINS - 1)
        mass = np.array([(ci == j).mean() for j in range(BINS)])
        polS[str(sd)] = {}
        for gi, g in enumerate(GAMMAS_C):
            m1, m0 = sharp_scores(Y, T, ci, BINS, g)
            pi = policies(np.nan_to_num(m1, nan=-1e9), np.nan_to_num(m0, nan=0.0), mass, cap)
            vals[gi].append(exact_cont(pi))
            polS[str(sd)]["%g" % g] = [round(float(v), 4) for v in pi]
    out_c["regimes"][reg] = {
        "mean": {"SharpIPW-O-X": [round(float(np.mean(vals[gi])), 4) for gi in range(len(GAMMAS_C))]},
        "sd": {"SharpIPW-O-X": [round(float(np.std(vals[gi])), 4) for gi in range(len(GAMMAS_C))]},
        "policy_by_seed": {"SharpIPW-O-X": polS}}
(HERE / "gstar_sharp_cont.json").write_text(json.dumps(out_c, indent=1))
g5c = GAMMAS_C.index(5.0)
print("continuous Sharp-O-X at Gamma*=5: uncap %.3f  cap %.3f"
      % (out_c["regimes"]["uncap"]["mean"]["SharpIPW-O-X"][g5c],
         out_c["regimes"]["cap"]["mean"]["SharpIPW-O-X"][g5c]))
print("saved gstar_sharp.json + gstar_sharp_cont.json")
