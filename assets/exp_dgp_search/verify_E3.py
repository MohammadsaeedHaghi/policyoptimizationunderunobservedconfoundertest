"""Verify E3 DGP works BOTH capped and uncapped: XX<OX<OW + large-Γ collapse. N=500, 5 seeds, Γ to 1000."""
import sys, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(ROOT)); import common
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
S = {}
for reg, suf, Cap in [("uncap", "uncapped", "Uncapped"), ("cap", "capped", "Capped")]:
    S[(reg, "IPW-X-X")] = L("methods/IPW-X-X/%s/ipw_x_x_%s.py" % (Cap, suf), "solve_ipw_x_x_%s" % suf)
    S[(reg, "DR-X-X")] = L("methods/DoublyRobust-X-X/%s/doublyrobust_x_x_%s.py" % (Cap, suf), "solve_doublyrobust_x_x_%s" % suf)
    S[(reg, "IPW-O-X")] = L("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf)
    S[(reg, "DR-O-X")] = L("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf)
    S[(reg, "IPW-O-W")] = L("methods/IPW-O-W/%s/ipw_o_w_%s.py" % (Cap, suf), "solve_ipw_o_w_%s" % suf)
    S[(reg, "DR-O-W")] = L("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf)
def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
P = dict(d=5.0, b=3.0, cs=4.0, cx=2.0, alpha=7.0, noise=0.9)
LV = np.round(np.linspace(-1, 1, 7), 6); K = 2; N = 500; SEEDS = list(range(5)); GAM = [2.0, 4.0, 8.0, 20.0, 100.0, 1000.0]
ES = 2 * sig(P["alpha"] * LV) - 1.0; eY0 = P["d"] * ES; eY1 = P["d"] * ES + P["b"] * LV
def val(pg): pg = np.asarray(pg, float); return float(np.mean(pg * eY1 + (1 - pg) * eY0))
orc = val((LV > 0).astype(float))
def gen(n, sd):
    rng = np.random.default_rng(sd); X = rng.choice(LV, size=n)
    Sv = np.where(rng.uniform(size=n) < sig(P["alpha"] * X), 1.0, -1.0)
    e = np.clip(sig(P["cs"] * Sv - P["cx"] * X), 0.02, 0.98); T = (rng.uniform(size=n) < e).astype(int)
    Y = np.where(T == 1, P["d"] * Sv + P["b"] * X, P["d"] * Sv) + rng.normal(0, P["noise"], size=n)
    return dict(X=X.reshape(-1, 1), T=T, Y=Y)
def tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])
print("E3 d=5 b=3 cs=4 cx=2 alpha=7 noise=0.9 | oracle=%.3f never=%.3f" % (orc, val(np.zeros(len(LV)))), flush=True)
t0 = time.time()
for reg in ["uncap", "cap"]:
    kw0 = {} if reg == "uncap" else {"cap": (1.0, 0.5)}
    acc = {m: [] for m in ["IPW-X-X", "DR-X-X", "IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]}
    for sd in SEEDS:
        o = gen(N, sd); w, _ = common.ipw_weights_from_data(o["X"], o["T"], K)
        mu = common.outcome_means(o["X"], o["T"], o["Y"], n_arms=K, cross_fit=True)
        Dm = common.pairwise_distance_matrix(o["X"]); eps = tuple(common.tight_epsilon(Dm, o["T"], w, K, is_distance=True, c_eps=1.0))
        acc["IPW-X-X"].append([val(tg(S[(reg, "IPW-X-X")](o["X"], o["T"], o["Y"], w, n_arms=K, discretize=False, **kw0)))] * len(GAM))
        acc["DR-X-X"].append([val(tg(S[(reg, "DR-X-X")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, discretize=False, **kw0)))] * len(GAM))
        for m in ["IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]: acc[m].append([])
        for g in GAM:
            acc["IPW-O-X"][-1].append(val(tg(S[(reg, "IPW-O-X")](o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, **kw0))))
            acc["DR-O-X"][-1].append(val(tg(S[(reg, "DR-O-X")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, **kw0))))
            acc["IPW-O-W"][-1].append(val(tg(S[(reg, "IPW-O-W")](o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0))))
            acc["DR-O-W"][-1].append(val(tg(S[(reg, "DR-O-W")](o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0))))
    M = {m: np.array(acc[m]).mean(0) for m in acc}
    xx = max(M["IPW-X-X"][0], M["DR-X-X"][0]); ox = max(M["IPW-O-X"].max(), M["DR-O-X"].max()); ow = max(M["IPW-O-W"].max(), M["DR-O-W"].max())
    print("\n=== %s (oracle %.3f) === XX=%.3f OX=%.3f OW=%.3f  %s" % (reg, orc, xx, ox, ow, "XX<OX<OW" if ox > xx + 0.02 and ow > ox else "(weak)"), flush=True)
    for m in ["IPW-X-X", "DR-X-X", "IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]:
        print("  %-9s byΓ=%s" % (m, np.round(M[m], 3).tolist()), flush=True)
print("\ndone %.1f min" % ((time.time() - t0) / 60), flush=True)
