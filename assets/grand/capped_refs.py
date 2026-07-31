#!/usr/bin/env python3
"""Cap-respecting naive and oracle references, per seed, on the SAME protocol as the solvers.

Bug this fixes: the runners compute their naive/oracle references as
    naive = 1[mu_hat(1) - mu_hat(0) > 0]        oracle = 1[CATE(x) > 0]
with NO capacity constraint, so the identical uncapped number was printed against every capped
row of every table. The robust methods were budget-limited to 30/40/50% of units while the
baselines they were compared to could treat anyone -- a comparison biased AGAINST the methods.

Protocol here is matched to the capped solvers, which impose (1/n) sum_i pi_i <= cap on the
TRAIN support and are then deployed off-support:
  * capped naive  -- rank TRAIN units by the cross-fit CATE-hat (the same `mu` the DR solvers
                     get), treat the top cap-fraction with a positive score, deploy by Shapley.
  * capped oracle -- the true ceiling: rank TEST units by the true marginal CATE and treat the
                     top cap-fraction with positive CATE. No estimation, no deployment.
No LPs, so this runs locally in seconds.

Output: assets/grand/capped_refs.json
"""
import json, sys, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
import common
from shapley import extract_support

CAPS = [("uncap", None), ("cap30", 0.3), ("cap40", 0.4), ("cap50", 0.5)]
CAMPS = {"gs": "assets/exp_gstar/dgp_cont.py",
         "km": "assets/exp_msmbench/dgp_g15.py",
         "kz": "assets/exp_kz18/dgp.py"}


def _shapley_fast(Xnew, sX, sp, block=200):
    if Xnew.ndim == 1: Xnew = Xnew.reshape(-1, 1)
    out = np.empty(Xnew.shape[0]); diag = np.arange(sX.shape[0])
    for s0 in range(0, Xnew.shape[0], block):
        xb = Xnew[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        S = dd[:, :, None] + dd[:, None, :]; S = np.where(S > 0, S, 1.0)
        A = (dd[:, None, :] * sp[None, :, None] + dd[:, :, None] * sp[None, None, :]) / S
        A[:, diag, diag] = sp[None, :]
        v = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0.0
        if ex.any(): v[ex] = sp[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = v
    return out


def _marginal_cate(d, x):
    """True marginal CATE at x (DGP's own cate() when present, else marginalise over S)."""
    x = np.asarray(x, float)
    if hasattr(d, "cate"): return np.asarray(d.cate(x), float)
    p = np.asarray(d.p_s1(x), float)
    hi = np.asarray(d.mu1(x, 1.0) - d.mu0(x, 1.0), float)
    lo = np.asarray(d.mu1(x, -1.0) - d.mu0(x, -1.0), float)
    return p * hi + (1.0 - p) * lo


def binned_cate_hat(X, T, Y, bins=15):
    """NONPARAMETRIC naive: per-arm bin means on quantile bins of the training x, contrasted.

    The linear naive (common.outcome_means -> LinearRegression) is well specified on gstar and
    roughly so on KZ18, but badly misspecified on KMZ, whose CATE is 2X + 2 - 4 sin(2X). A naive
    baseline that is LESS flexible than our own Lipschitz policy class flatters us wherever the
    truth is linear and handicaps us wherever it is not, so we report both. This estimator is
    the Gamma = 1 limit of the sharp baseline (at Gamma = 1 the MSM box collapses to a point and
    the sharp score reduces to the binned contrast), which keeps the two comparable.

    Bins with no data in one arm fall back to that arm's global mean.
    """
    x = np.asarray(X, float).ravel()
    T = np.asarray(T).astype(int).ravel(); Y = np.asarray(Y, float).ravel()
    ed = np.quantile(x, np.linspace(0, 1, bins + 1)); ed[0] -= 1e-9; ed[-1] += 1e-9
    ci = np.clip(np.digitize(x, ed) - 1, 0, bins - 1)
    g1 = Y[T == 1].mean() if (T == 1).any() else 0.0
    g0 = Y[T == 0].mean() if (T == 0).any() else 0.0
    m1 = np.full(bins, g1); m0 = np.full(bins, g0)
    for j in range(bins):
        s = ci == j
        if (s & (T == 1)).any(): m1[j] = Y[s & (T == 1)].mean()
        if (s & (T == 0)).any(): m0[j] = Y[s & (T == 0)].mean()
    return (m1 - m0)[ci]


def top_cap(score, cap):
    """0/1 policy treating the top cap-fraction of units with a POSITIVE score."""
    n = len(score); pi = np.zeros(n)
    if cap is None: return (score > 0).astype(float)
    k = int(np.floor(cap * n))
    if k <= 0: return pi
    order = np.argsort(-np.asarray(score, float))[:k]
    order = [i for i in order if score[i] > 0]
    pi[order] = 1.0
    return pi


def main(seeds=10, n=200, nte=4000):
    out = {}
    for camp, rel in CAMPS.items():
        sp = importlib.util.spec_from_file_location("cr_" + camp, ROOT / rel)
        d = importlib.util.module_from_spec(sp); sys.modules["cr_" + camp] = d; sp.loader.exec_module(d)
        K = d.K
        acc = {c: {"naive": [], "naive_np": [], "oracle": [], "never": [],
                   "treat_frac": [], "treat_frac_np": []} for c, _ in CAPS}
        for sd in range(seeds):
            obs, _ = d.generate(n, sd); X, T, Y = obs["X"], obs["T"], obs["Y"]
            te, ft = d.generate(nte, sd + 1000); Xte = te["X"]; Y1t, Y0t = ft["Y1"], ft["Y0"]
            mu = common.outcome_means(X, T, Y, n_arms=K, cross_fit=True)
            score = mu[:, 1] - mu[:, 0]                       # LINEAR naive CATE-hat on train
            score_np = binned_cate_hat(X, T, Y)               # NONPARAMETRIC (binned) naive
            cate_te = _marginal_cate(d, Xte.ravel())          # truth on test
            for cname, cap in CAPS:
                pin = top_cap(score, cap)                     # cap imposed on TRAIN, as the solvers do
                sX, spv = extract_support(X, pin)
                dep = _shapley_fast(np.asarray(Xte, float), np.asarray(sX, float), np.asarray(spv, float).ravel())
                acc[cname]["naive"].append(float(np.mean(dep * Y1t + (1 - dep) * Y0t)))
                acc[cname]["treat_frac"].append(float(np.mean(dep)))
                pnp = top_cap(score_np, cap)                  # nonparametric naive, same protocol
                sXn, spn = extract_support(X, pnp)
                depn = _shapley_fast(np.asarray(Xte, float), np.asarray(sXn, float), np.asarray(spn, float).ravel())
                acc[cname]["naive_np"].append(float(np.mean(depn * Y1t + (1 - depn) * Y0t)))
                acc[cname]["treat_frac_np"].append(float(np.mean(depn)))
                po = top_cap(cate_te, cap)                    # true capped ceiling, on test
                acc[cname]["oracle"].append(float(np.mean(po * Y1t + (1 - po) * Y0t)))
                acc[cname]["never"].append(float(np.mean(Y0t)))
        out[camp] = {c: {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
                         for k, v in acc[c].items()} for c, _ in CAPS}
        print("== %s (%d seeds, n=%d)" % (camp, seeds, n))
        for c, _ in CAPS:
            r = out[camp][c]
            print("   %-7s oracle %7.3f | naive-linear %7.3f (%.2f) | naive-nonpar %7.3f (%.2f) | never %7.3f"
                  % (c, r["oracle"][0], r["naive"][0], r["treat_frac"][0],
                     r["naive_np"][0], r["treat_frac_np"][0], r["never"][0]))
    (HERE / "capped_refs.json").write_text(json.dumps({"seeds": seeds, "n": n, "refs": out}, indent=1))
    print("\nsaved assets/grand/capped_refs.json")


if __name__ == "__main__":
    main()
