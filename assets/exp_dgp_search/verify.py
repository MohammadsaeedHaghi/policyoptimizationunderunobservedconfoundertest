"""Verify a chosen discrete DGP at N=500: per-seed family bests + win-rate (fraction of seeds with OW>OX>XX),
and the ordering at each fixed Γ (so the win is not a cherry-picked peak). Exact analytic evaluation."""
import sys, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(ROOT)); import common
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped")
drxx  = L("methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py", "solve_doublyrobust_x_x_capped")
ipwox = L("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped")
drox  = L("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped")
ipwow = L("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped")
drow  = L("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped")
def sig(z): return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(z, float), -40, 40)))

def make(p):
    LV = np.round(np.linspace(-1, 1, p["levels"]), 6)
    def gen(n, seed):
        rng = np.random.default_rng(seed); X = rng.choice(LV, size=n)
        S = np.where(rng.uniform(size=n) < sig(p["alpha"] * X), 1.0, -1.0)
        e = np.clip(sig(p["cs"] * S - p["cx"] * X), 0.02, 0.98); T = (rng.uniform(size=n) < e).astype(int)
        m0, m1 = p["d"] * S, p["d"] * S + X
        Y = np.where(T == 1, m1, m0) + rng.normal(0, p["noise"], size=n)
        return dict(X=X.reshape(-1, 1), T=T, Y=Y)
    ES = 2 * sig(p["alpha"] * LV) - 1.0
    return dict(LV=LV, gen=gen, eY0=p["d"] * ES, eY1=p["d"] * ES + LV, oracle=(LV > 0).astype(float))

def run(p, N, seeds, gammas, cap=(1.0, 0.5)):
    dgp = make(p); LV = dgp["LV"]; K = 2
    def tg(res):
        s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
        pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
        idx = {round(float(c), 6): i for i, c in enumerate(lv)}
        return np.array([pol[idx[round(float(v), 6)]] for v in LV])
    def val(pg): pg = np.asarray(pg, float); return float(np.mean(pg * dgp["eY1"] + (1 - pg) * dgp["eY0"]))
    orc = val(dgp["oracle"]); rows = {m: [] for m in ["IPW-X-X", "DR-X-X", "IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]}
    for sd in seeds:
        d = dgp["gen"](N, sd); w, _ = common.ipw_weights_from_data(d["X"], d["T"], K)
        mu = common.outcome_means(d["X"], d["T"], d["Y"], n_arms=K, cross_fit=True)
        Dm = common.pairwise_distance_matrix(d["X"]); eps = tuple(common.tight_epsilon(Dm, d["T"], w, K, is_distance=True, c_eps=1.0))
        rows["IPW-X-X"].append([val(tg(ipwxx(d["X"], d["T"], d["Y"], w, n_arms=K, cap=cap, discretize=False)))] * len(gammas))
        rows["DR-X-X"].append([val(tg(drxx(d["X"], d["T"], d["Y"], w, mu, n_arms=K, cap=cap, discretize=False)))] * len(gammas))
        for nm in ["IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]: rows[nm].append([])
        for g in gammas:
            rows["IPW-O-X"][-1].append(val(tg(ipwox(d["X"], d["T"], d["Y"], w, n_arms=K, Gamma=g, cap=cap, discretize=False))))
            rows["DR-O-X"][-1].append(val(tg(drox(d["X"], d["T"], d["Y"], w, mu, n_arms=K, Gamma=g, cap=cap, discretize=False))))
            rows["IPW-O-W"][-1].append(val(tg(ipwow(d["X"], d["T"], d["Y"], w, n_arms=K, Gamma=g, cap=cap, discretize=False, zscore=False, epsilon=eps))))
            rows["DR-O-W"][-1].append(val(tg(drow(d["X"], d["T"], d["Y"], w, mu, n_arms=K, Gamma=g, cap=cap, discretize=False, zscore=False, epsilon=eps))))
    R = {m: np.array(rows[m]) for m in rows}  # (seeds, gammas)
    return R, orc

P = dict(levels=7, alpha=6.0, cs=2.0, cx=2.0, d=2.5, noise=0.6)   # v3 (locked)
N = 700; SEEDS = list(range(8)); GAMMAS = [2.0, 3.0, 5.0, 8.0, 12.0]
t0 = time.time()
for cap in [(1.0, 0.5)]:
    R, orc = run(P, N, SEEDS, GAMMAS, cap=cap)
    # per-seed win-rate WITHIN each estimator family, using each method's OWN best-Γ
    def best(m, s): return R[m][s].max()
    dr = sum(best("DR-O-W", s) > best("DR-O-X", s) > best("DR-X-X", s) for s in range(len(SEEDS)))
    ip = sum(best("IPW-O-W", s) > best("IPW-O-X", s) > best("IPW-X-X", s) for s in range(len(SEEDS)))
    print("=== cap=%s | N=%d seeds=%d oracle=%.3f ===" % (cap, N, len(SEEDS), orc))
    print("per-seed win-rate  DR(O-W>O-X>X-X)=%d/%d   IPW(O-W>O-X>X-X)=%d/%d" % (dr, len(SEEDS), ip, len(SEEDS)))
    for m in ["IPW-X-X", "DR-X-X", "IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]:
        mu = R[m].mean(0); sd = R[m].std(0); print("  %-9s mean=%s sd=%s" % (m, np.round(mu, 3).tolist(), np.round(sd, 3).tolist()))
    print("  DR fixed-Γ %s:" % list(map(int, GAMMAS)))
    for gi, g in enumerate(GAMMAS):
        a, b, c = R["DR-X-X"].mean(0)[gi], R["DR-O-X"].mean(0)[gi], R["DR-O-W"].mean(0)[gi]
        print("    Γ=%g  XX=%.3f O-X=%.3f O-W=%.3f %s" % (g, a, b, c, "OW>OX>XX" if c > b > a else "x"))
print("\ndone %.1f min" % ((time.time() - t0) / 60))
