"""Refine around E_balance: want UNCAPPED big-margin XX<OX<OW AND both OX & OW degrade at large Γ (to 1000).
Stronger selection/confounding to fool XX more. N=400, 3 seeds, exact value, uncapped."""
import sys, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(ROOT)); import common
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped")
drxx  = L("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped")
ipwox = L("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
drox  = L("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped")
ipwow = L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
drow  = L("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")
def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))
LV = np.round(np.linspace(-1, 1, 7), 6); K = 2; N = 400; SEEDS = [0, 1, 2]; GAM = [2.0, 4.0, 8.0, 20.0, 100.0, 1000.0]
def make(p):
    def gen(n, sd):
        rng = np.random.default_rng(sd); X = rng.choice(LV, size=n)
        S = np.where(rng.uniform(size=n) < sig(p["alpha"] * X), 1.0, -1.0)
        e = np.clip(sig(p["cs"] * S - p["cx"] * X), 0.02, 0.98); T = (rng.uniform(size=n) < e).astype(int)
        m0, m1 = p["d"] * S, p["d"] * S + p["b"] * X
        Y = np.where(T == 1, m1, m0) + rng.normal(0, p["noise"], size=n)
        return dict(X=X.reshape(-1, 1), T=T, Y=Y)
    ES = 2 * sig(p["alpha"] * LV) - 1.0
    return gen, p["d"] * ES, p["d"] * ES + p["b"] * LV
def tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])
CONFIGS = {
 "E":  dict(d=3.5, b=3.0, cs=2.5, cx=2.0, alpha=6.0, noise=0.8),
 "E2": dict(d=4.0, b=3.0, cs=3.5, cx=2.0, alpha=7.0, noise=0.8),
 "E3": dict(d=5.0, b=3.0, cs=4.0, cx=2.0, alpha=7.0, noise=0.9),
 "E4": dict(d=4.5, b=2.5, cs=3.5, cx=2.0, alpha=7.0, noise=0.8),
}
t0 = time.time()
for name, p in CONFIGS.items():
    gen, eY0, eY1 = make(p)
    def val(pg): pg = np.asarray(pg, float); return float(np.mean(pg * eY1 + (1 - pg) * eY0))
    orc = val((LV > 0).astype(float))
    acc = {m: [] for m in ["IPW-X-X", "DR-X-X", "IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]}
    for sd in SEEDS:
        o = gen(N, sd); w, _ = common.ipw_weights_from_data(o["X"], o["T"], K)
        mu = common.outcome_means(o["X"], o["T"], o["Y"], n_arms=K, cross_fit=True)
        Dm = common.pairwise_distance_matrix(o["X"]); eps = tuple(common.tight_epsilon(Dm, o["T"], w, K, is_distance=True, c_eps=1.0))
        acc["IPW-X-X"].append([val(tg(ipwxx(o["X"], o["T"], o["Y"], w, n_arms=K, discretize=False)))] * len(GAM))
        acc["DR-X-X"].append([val(tg(drxx(o["X"], o["T"], o["Y"], w, mu, n_arms=K, discretize=False)))] * len(GAM))
        for m in ["IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]: acc[m].append([])
        for g in GAM:
            acc["IPW-O-X"][-1].append(val(tg(ipwox(o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False))))
            acc["DR-O-X"][-1].append(val(tg(drox(o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False))))
            acc["IPW-O-W"][-1].append(val(tg(ipwow(o["X"], o["T"], o["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps))))
            acc["DR-O-W"][-1].append(val(tg(drow(o["X"], o["T"], o["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps))))
    M = {m: np.array(acc[m]).mean(0) for m in acc}
    xx = max(M["IPW-X-X"][0], M["DR-X-X"][0]); ox = max(M["IPW-O-X"].max(), M["DR-O-X"].max()); ow = max(M["IPW-O-W"].max(), M["DR-O-W"].max())
    drox_drop = M["DR-O-X"].max() - M["DR-O-X"][-1]; drow_drop = M["DR-O-W"].max() - M["DR-O-W"][-1]
    flag = " XX<OX<OW" if (ox > xx + 0.02 and ow > ox + 0.005) else ""
    print("[%s] orc=%.2f | XX=%.3f OX=%.3f OW=%.3f | DR-O-X drop=%.2f DR-O-W drop=%.2f%s" % (name, orc, xx, ox, ow, drox_drop, drow_drop, flag), flush=True)
    print("     DR  X-X=%.3f O-X=%s O-W=%s" % (M["DR-X-X"][0], np.round(M["DR-O-X"], 3).tolist(), np.round(M["DR-O-W"], 3).tolist()), flush=True)
print("done %.1f min" % ((time.time() - t0) / 60), flush=True)
