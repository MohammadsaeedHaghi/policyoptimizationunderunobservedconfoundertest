#!/usr/bin/env python3
"""Sharp-O-X baseline across EVERY cap budget and EVERY confounding strength, all 3 benchmarks.

Fills the two gaps the earlier per-campaign sharp scripts left:
  * cap 40% and 50% (previously only 30%), so the capped tables have a sharp column throughout;
  * the Gamma*-strength ladder for gstar and KZ18 (previously only KMZ had one).

Same two-point sharp adversary as assets/exp_gstar/run_sharp.py (MSM box + the sharpness
equality E[W | T=t, cell] = 1/e_hat, solved in closed form by sorting) -- license-free, so this
runs locally while the solver wave occupies the cluster. Protocol matched to the grand campaign:
n = 200 train, 10 seeds, evaluation on a fixed 200k-unit draw with known counterfactuals.

Output: assets/grand/sharp_all.json
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

def _load(p, n):
    sp = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m; sp.loader.exec_module(m); return m

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
        e = float(np.clip(T[m].mean(), 0.02, 0.98))
        yt, yc = Y[m & (T == 1)], Y[m & (T == 0)]
        m1[j] = sharp_min_arm(yt, 1.0 / e, Gamma) * (len(yt) / max(m.sum(), 1))
        m0[j] = sharp_min_arm(yc, 1.0 / (1 - e), Gamma) * (len(yc) / max(m.sum(), 1))
    return m1, m0

def _cate(d, x):
    """Marginal CATE at x. Uses the DGP's own cate() when it exposes one (msmbench, kz18);
    otherwise marginalises mu1 - mu0 over the hidden S given x (gstar family)."""
    if hasattr(d, "cate"):
        return float(np.asarray(d.cate(x)).ravel()[0])
    p = float(np.asarray(d.p_s1(x)).ravel()[0])
    hi = float(np.asarray(d.mu1(x, 1.0) - d.mu0(x, 1.0)).ravel()[0])
    lo = float(np.asarray(d.mu1(x, -1.0) - d.mu0(x, -1.0)).ravel()[0])
    return p * hi + (1.0 - p) * lo


def policy(score, mass, cap=None):
    if cap is None: return (np.asarray(score) > 0).astype(float)
    pi = np.zeros(len(score)); rem = cap
    for j in sorted(range(len(score)), key=lambda i: -score[i]):
        if score[j] <= 0 or rem <= 1e-12: break
        take = min(1.0, rem / max(mass[j], 1e-12)); pi[j] = take; rem -= take * mass[j]
    return pi

NTRAIN, NTEST, BINS = 200, 200000, 15
SEEDS = list(range(10))
CAPS = [("uncap", None), ("cap30", 0.3), ("cap40", 0.4), ("cap50", 0.5)]

# campaign -> (label, dgp path, Gamma grid, matched Gamma)
CAMPAIGNS = {
    "gs": [("L=2", "exp_gstar/dgp_l2_cont.py", 2.0), ("L=3", "exp_gstar/dgp_l3_cont.py", 3.0),
           ("L=5 (main)", "exp_gstar/dgp_cont.py", 5.0), ("L=8", "exp_gstar/dgp_l8_cont.py", 8.0)],
    "km": [("e^0.5", "exp_msmbench/dgp_g05.py", 1.6487), ("e^1.0", "exp_msmbench/dgp_g10.py", 2.7183),
           ("e^1.5 (main)", "exp_msmbench/dgp_g15.py", 4.4817), ("e^2.0", "exp_msmbench/dgp_g20.py", 7.3891)],
    "kz": [("e^0.5", "exp_kz18/dgp_g05.py", 1.6487), ("e^1.0", "exp_kz18/dgp_g10.py", 2.7183),
           ("e^1.5 (main)", "exp_kz18/dgp.py", 4.4817), ("e^2.0", "exp_kz18/dgp_g20.py", 7.3891)],
}
GGRID = {"gs": [1, 2, 3, 5, 8, 12, 16, 24, 32],
         "km": [1, 1.6487, 2.7183, 4.4817, 7.3891, 12, 16, 24, 32],
         "kz": [1, 1.6487, 2.7183, 4.4817, 7.3891, 12, 16, 24, 32, 40]}

out = {"n_train": NTRAIN, "n_test": NTEST, "bins": BINS, "seeds": SEEDS, "campaigns": {}}
for camp, arms in CAMPAIGNS.items():
    gg = GGRID[camp]
    out["campaigns"][camp] = {"gammas": gg, "arms": {}}
    for lab, rel, gstar in arms:
        d = _load(ROOT / "assets" / rel, "sh_%s_%s" % (camp, rel.replace("/", "_")))
        te, ft = d.generate(NTEST, 999)
        xte = te["X"].ravel(); Y1, Y0 = ft["Y1"], ft["Y0"]
        edges = np.quantile(xte, np.linspace(0, 1, BINS + 1)); edges[0] -= 1e-9; edges[-1] += 1e-9
        cte = np.clip(np.digitize(xte, edges) - 1, 0, BINS - 1)
        mass = np.array([(cte == j).mean() for j in range(BINS)])
        def val(pb):
            p = np.asarray(pb, float)[cte]
            return float(np.mean(p * Y1 + (1 - p) * Y0))
        # references on the same draw
        ob, fb = d.generate(NTEST, 12345)
        xb, Tb, Yb = ob["X"].ravel(), ob["T"], ob["Y"]
        cb = np.clip(np.digitize(xb, edges) - 1, 0, BINS - 1)
        hat = np.zeros(BINS)
        for j in range(BINS):
            m = cb == j; yt, yc = Yb[m & (Tb == 1)], Yb[m & (Tb == 0)]
            if len(yt) > 5 and len(yc) > 5: hat[j] = yt.mean() - yc.mean()
        cate_b = np.array([_cate(d, 0.5 * (edges[j] + edges[j + 1])) for j in range(BINS)])
        node = {"gstar": gstar, "never": float(np.mean(Y0)), "all": float(np.mean(Y1)),
                "corr_xS": float(np.corrcoef(xb, fb["S"])[0, 1]), "regimes": {}, "refs": {}}
        for cname, cap in CAPS:
            node["refs"][cname] = {"oracle": val(policy(cate_b, mass, cap)),
                                   "naive": val(policy(hat, mass, cap))}
            vals = {gi: [] for gi in range(len(gg))}
            for sd in SEEDS:
                obs, _ = d.generate(NTRAIN, sd)
                X, T, Y = obs["X"].ravel(), obs["T"], obs["Y"]
                # LEAK FIX: bins come from THIS SEED'S TRAINING x, never from the test draw.
                # Deriving them from the 200k evaluation sample leaked the test covariate
                # distribution into the learned policy and quietly advantaged this baseline
                # over the solvers, which see only training data.
                ed_tr = np.quantile(X, np.linspace(0, 1, BINS + 1))
                ed_tr[0] -= 1e-9; ed_tr[-1] += 1e-9
                ci = np.clip(np.digitize(X, ed_tr) - 1, 0, BINS - 1)
                cte_s = np.clip(np.digitize(xte, ed_tr) - 1, 0, BINS - 1)
                mass_s = np.array([(cte_s == j).mean() for j in range(BINS)])
                def val_s(pb, _c=cte_s):
                    pv = np.asarray(pb, float)[_c]
                    return float(np.mean(pv * Y1 + (1 - pv) * Y0))
                for gi, g in enumerate(gg):
                    m1, m0 = sharp_scores(Y, T, ci, BINS, float(g))
                    sc = np.nan_to_num(m1, nan=-1e9) - np.nan_to_num(m0, nan=0.0)
                    vals[gi].append(val_s(policy(sc, mass_s, cap)))
            node["regimes"][cname] = {
                "mean": [round(float(np.mean(vals[gi])), 4) for gi in range(len(gg))],
                "sd": [round(float(np.std(vals[gi])), 4) for gi in range(len(gg))]}
        out["campaigns"][camp]["arms"][lab] = node
        bi = int(np.argmax(node["regimes"]["uncap"]["mean"]))
        print("[%s %-12s] Gamma*=%-7.3f uncap best %.3f at G=%-7s (matched %.3f) | oracle %.3f naive %.3f"
              % (camp, lab, gstar, node["regimes"]["uncap"]["mean"][bi], gg[bi],
                 node["regimes"]["uncap"]["mean"][min(range(len(gg)), key=lambda i: abs(gg[i] - gstar))],
                 node["refs"]["uncap"]["oracle"], node["refs"]["uncap"]["naive"]), flush=True)

(HERE / "sharp_all_trainbins.json").write_text(json.dumps(out, indent=1))
print("saved assets/grand/sharp_all.json")
