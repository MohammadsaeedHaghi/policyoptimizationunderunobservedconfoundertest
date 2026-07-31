#!/usr/bin/env python3
"""Corrected Sharp-O-X + naive baselines for the gstar report, at the report's OWN sample sizes.

WHY THIS REPLACES run_sharp.py
------------------------------
The gstar report's sharp column was measured with a protocol that handicapped it:
  * EQUALLY SPACED bin edges (run_sharp.py:108, `np.linspace(-1, 1, BINS+1)`) instead of
    quantile bins from the training sample, and
  * analytic bin-mean evaluation rather than the test-draw evaluation every other method gets.
Re-measuring with the current protocol at the report's own n = 400 gives sharp 0.567 (5 seeds) /
0.586 (10 seeds) on continuous-uncapped, versus the 0.502 the report prints -- i.e. sharp was
ALREADY AHEAD of IPW-O-W's 0.549 there and the report shows it losing. That is a reporting
error, not a finding, and this script fixes it.

Also computes the naive references the report needs and did not have:
  * cap-RESPECTING naive (the runner's naive ignores the capacity constraint, so the same
    uncapped number was printed against every capped row), and
  * a NONPARAMETRIC (binned) naive alongside the linear one -- the linear naive is well specified
    on gstar and its ranking correlates 0.988 with the true CATE, better than a perfectly
    estimated confounded naive would (0.736), so quoting it alone flatters the baseline.

Discrete uses the 7 LEVELS as its own cells (no binning choice to make) and the DGP's exact_value;
continuous uses 15 quantile bins from TRAIN and a 200k test draw. Per-bin policies are saved so
the report's pi(x) viewers can overlay sharp.

Output: assets/exp_gstar/gstar_sharp_v2.json
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "assets" / "grand"))
import common
from run_sharp_all import sharp_scores, policy

BINS = 15
NTE = 200000
PGRID = np.linspace(-1.0, 1.0, 41)
N_DISC, N_CONT = 600, 400          # the report's own sample sizes
SEEDS = list(range(5))             # the report's own seed count
GAMMAS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]


def _load(rel, name):
    sp = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(sp); sys.modules[name] = m; sp.loader.exec_module(m)
    return m


def naive_refs_cont(d, n, caps):
    """Cap-respecting naive (linear and binned) on the continuous DGP, solver protocol."""
    te, ft = d.generate(NTE, 999); xte = te["X"].ravel(); Y1, Y0 = ft["Y1"], ft["Y0"]
    out = {}
    for cname, cap in caps:
        lin, nonp, orc = [], [], []
        for sd in SEEDS:
            obs, _ = d.generate(n, sd); X, T, Y = obs["X"], obs["T"], obs["Y"]
            x = X.ravel()
            mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
            ed = np.quantile(x, np.linspace(0, 1, BINS + 1)); ed[0] -= 1e-9; ed[-1] += 1e-9
            ci = np.clip(np.digitize(x, ed) - 1, 0, BINS - 1)
            cte = np.clip(np.digitize(xte, ed) - 1, 0, BINS - 1)
            b1 = np.zeros(BINS)
            for j in range(BINS):
                m = ci == j; yt, yc = Y[m & (T == 1)], Y[m & (T == 0)]
                if len(yt) > 1 and len(yc) > 1: b1[j] = yt.mean() - yc.mean()
            for score_bins, acc in ((None, lin), (b1, nonp)):
                if score_bins is None:                       # linear naive, ranked per unit
                    sc = mu[:, 1] - mu[:, 0]
                    if cap is None: pol_u = (sc > 0).astype(float)
                    else:
                        pol_u = np.zeros(len(sc)); k = int(np.floor(cap * len(sc)))
                        for i in np.argsort(-sc)[:k]:
                            if sc[i] > 0: pol_u[i] = 1.0
                    cellpi = np.array([pol_u[ci == j].mean() if (ci == j).any() else 0.0 for j in range(BINS)])
                else:
                    mass = np.array([(cte == j).mean() for j in range(BINS)])
                    cellpi = policy(score_bins, mass, cap)
                p = cellpi[cte]
                acc.append(float(np.mean(p * Y1 + (1 - p) * Y0)))
            ctrue = np.array([d.cate(0.5 * (ed[j] + ed[j + 1])) if hasattr(d, "cate") else
                              (d.p_s1(0.5 * (ed[j] + ed[j + 1])) * (d.mu1(0.5 * (ed[j] + ed[j + 1]), 1.0) - d.mu0(0.5 * (ed[j] + ed[j + 1]), 1.0))
                               + (1 - d.p_s1(0.5 * (ed[j] + ed[j + 1]))) * (d.mu1(0.5 * (ed[j] + ed[j + 1]), -1.0) - d.mu0(0.5 * (ed[j] + ed[j + 1]), -1.0)))
                              for j in range(BINS)], dtype=float).ravel()
            mass = np.array([(cte == j).mean() for j in range(BINS)])
            po = policy(ctrue, mass, cap)[cte]
            orc.append(float(np.mean(po * Y1 + (1 - po) * Y0)))
        out[cname] = {"naive_linear": round(float(np.mean(lin)), 4),
                      "naive_binned": round(float(np.mean(nonp)), 4),
                      "oracle": round(float(np.mean(orc)), 4)}
    return out


out = {"note": "corrected protocol: quantile bins from TRAIN, test-draw evaluation",
       "n_disc": N_DISC, "n_cont": N_CONT, "seeds": SEEDS, "gammas": GAMMAS, "bins": BINS}

# ---------------- DISCRETE: the 7 LEVELS are the cells; exact_value evaluation ----------------
dd = _load("assets/exp_gstar/dgp.py", "gs_disc_v2")
lv = np.asarray(dd.LEVELS, float); nlev = len(lv)
disc = {}
for cname, cap in (("uncap", None), ("cap30", 0.3), ("cap40", 0.4), ("cap50", 0.5)):
    vals = {g: [] for g in GAMMAS}; pol0 = {}
    for sd in SEEDS:
        obs, _ = dd.generate(N_DISC, sd); X, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
        ci = np.array([int(np.argmin(np.abs(lv - v))) for v in X])
        mass = np.full(nlev, 1.0 / nlev)
        for g in GAMMAS:
            m1, m0 = sharp_scores(Y, T, ci, nlev, g)
            sc = np.nan_to_num(m1, nan=-1e9) - np.nan_to_num(m0, nan=0.0)
            pi = policy(sc, mass, cap)
            vals[g].append(dd.exact_value(pi))
            if sd == 0: pol0["%g" % g] = [round(float(v), 4) for v in pi]
    disc[cname] = {"mean": [round(float(np.mean(vals[g])), 4) for g in GAMMAS],
                   "sd": [round(float(np.std(vals[g])), 4) for g in GAMMAS],
                   "policy_seed0": pol0}
    print("[discrete %-6s] sharp at Gamma*=5: %.3f | best %.3f"
          % (cname, disc[cname]["mean"][GAMMAS.index(5.0)], max(disc[cname]["mean"])), flush=True)
out["discrete"] = disc
out["levels"] = [round(float(v), 6) for v in lv]

# ---------------- CONTINUOUS: 15 quantile bins from TRAIN, test-draw evaluation ----------------
dc = _load("assets/exp_gstar/dgp_cont.py", "gs_cont_v2")
te, ft = dc.generate(NTE, 999); xte = te["X"].ravel(); Y1, Y0 = ft["Y1"], ft["Y0"]
cont = {}
for cname, cap in (("uncap", None), ("cap30", 0.3)):
    vals = {g: [] for g in GAMMAS}; polg = {}
    for sd in SEEDS:
        obs, _ = dc.generate(N_CONT, sd); X, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
        ed = np.quantile(X, np.linspace(0, 1, BINS + 1)); ed[0] -= 1e-9; ed[-1] += 1e-9
        ci = np.clip(np.digitize(X, ed) - 1, 0, BINS - 1)
        cte = np.clip(np.digitize(xte, ed) - 1, 0, BINS - 1)
        cpg = np.clip(np.digitize(PGRID, ed) - 1, 0, BINS - 1)
        mass = np.array([(cte == j).mean() for j in range(BINS)])
        for g in GAMMAS:
            m1, m0 = sharp_scores(Y, T, ci, BINS, g)
            sc = np.nan_to_num(m1, nan=-1e9) - np.nan_to_num(m0, nan=0.0)
            cellpi = policy(sc, mass, cap)
            p = cellpi[cte]
            vals[g].append(float(np.mean(p * Y1 + (1 - p) * Y0)))
            if sd == 0: polg["%g" % g] = [round(float(v), 4) for v in cellpi[cpg]]
    cont[cname] = {"mean": [round(float(np.mean(vals[g])), 4) for g in GAMMAS],
                   "sd": [round(float(np.std(vals[g])), 4) for g in GAMMAS],
                   "policy_grid_seed0": polg}
    print("[continuous %-6s] sharp at Gamma*=5: %.3f | best %.3f"
          % (cname, cont[cname]["mean"][GAMMAS.index(5.0)], max(cont[cname]["mean"])), flush=True)
out["continuous"] = cont
out["policy_grid"] = [round(float(v), 4) for v in PGRID]

out["naive_cont"] = naive_refs_cont(dc, N_CONT, (("uncap", None), ("cap30", 0.3)))
print("continuous naive refs:", json.dumps(out["naive_cont"]), flush=True)
(HERE / "gstar_sharp_v2.json").write_text(json.dumps(out, indent=1))
print("saved assets/exp_gstar/gstar_sharp_v2.json")
