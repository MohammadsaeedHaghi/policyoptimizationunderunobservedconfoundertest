"""Fast DGP search: find a simple DISCRETE-X DGP where OW family > OX family > XX family (capped).
Outcomes continuous Gaussian; X on a discrete grid; S in {-1,+1} correlated with X (P(S=1|X)=sigma(alpha X)).
Evaluation is EXACT (analytic expected value on the uniform grid, no test sampling noise) -> only training-sample
noise remains, averaged over seeds. Run: python3 search.py
"""
import sys, importlib.util, json, time
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
    def pS1(X): return sig(p["alpha"] * np.asarray(X, float))           # P(S=+1|X)
    def e(X, S): return np.clip(sig(p["cs"] * np.asarray(S, float) - p["cx"] * np.asarray(X, float)), 0.02, 0.98)
    def mu0(X, S): return p["d"] * np.asarray(S, float)
    def mu1(X, S): return p["d"] * np.asarray(S, float) + np.asarray(X, float)
    def gen(n, seed):
        rng = np.random.default_rng(seed)
        X = rng.choice(LV, size=n)
        S = np.where(rng.uniform(size=n) < pS1(X), 1.0, -1.0)
        T = (rng.uniform(size=n) < e(X, S)).astype(int)
        m0, m1 = mu0(X, S), mu1(X, S)
        Y = np.where(T == 1, m1, m0) + rng.normal(0, p["noise"], size=n)
        return dict(X=X.reshape(-1, 1), T=T, Y=Y, S=S)
    # exact per-level expected outcomes (uniform X on grid)
    ps1 = pS1(LV); ES = 2 * ps1 - 1.0                                    # E[S|X]
    eY0 = p["d"] * ES; eY1 = p["d"] * ES + LV                            # E[Y(t)|X]
    return dict(LV=LV, gen=gen, eY0=eY0, eY1=eY1, oracle=(LV > 0).astype(float))

def value(pol_grid, dgp):                                                # exact expected value, X uniform on grid
    pol = np.asarray(pol_grid, float)
    return float(np.mean(pol * dgp["eY1"] + (1 - pol) * dgp["eY0"]))

def run(p, N, seeds, gammas, cap=(1.0, 0.5)):
    dgp = make(p); LV = dgp["LV"]; K = 2
    def togrid(res):
        s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
        pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
        idx = {round(float(c), 6): i for i, c in enumerate(lv)}
        return np.array([pol[idx[round(float(v), 6)]] for v in LV])
    acc = {m: [] for m in ["IPW-X-X", "DR-X-X", "IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]}
    orc = value(dgp["oracle"], dgp)
    for sd in seeds:
        d = dgp["gen"](N, sd)
        w, _ = common.ipw_weights_from_data(d["X"], d["T"], K)
        mu = common.outcome_means(d["X"], d["T"], d["Y"], n_arms=K, cross_fit=True)
        Dm = common.pairwise_distance_matrix(d["X"]); eps = tuple(common.tight_epsilon(Dm, d["T"], w, K, is_distance=True, c_eps=1.0))
        acc["IPW-X-X"].append([value(togrid(ipwxx(d["X"], d["T"], d["Y"], w, n_arms=K, cap=cap, discretize=False)), dgp)] * len(gammas))
        acc["DR-X-X"].append([value(togrid(drxx(d["X"], d["T"], d["Y"], w, mu, n_arms=K, cap=cap, discretize=False)), dgp)] * len(gammas))
        for nm in ["IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]: acc[nm].append([])
        for g in gammas:
            acc["IPW-O-X"][-1].append(value(togrid(ipwox(d["X"], d["T"], d["Y"], w, n_arms=K, Gamma=g, cap=cap, discretize=False)), dgp))
            acc["DR-O-X"][-1].append(value(togrid(drox(d["X"], d["T"], d["Y"], w, mu, n_arms=K, Gamma=g, cap=cap, discretize=False)), dgp))
            acc["IPW-O-W"][-1].append(value(togrid(ipwow(d["X"], d["T"], d["Y"], w, n_arms=K, Gamma=g, cap=cap, discretize=False, zscore=False, epsilon=eps)), dgp))
            acc["DR-O-W"][-1].append(value(togrid(drow(d["X"], d["T"], d["Y"], w, mu, n_arms=K, Gamma=g, cap=cap, discretize=False, zscore=False, epsilon=eps)), dgp))
    mean = {m: np.array(acc[m]).mean(0) for m in acc}                    # (len gammas,)
    return mean, orc, eps

CONFIGS = {
  "v1": dict(levels=9, alpha=5.0, cs=1.5, cx=2.0, d=2.0, noise=0.5),
  "v2": dict(levels=9, alpha=6.0, cs=2.0, cx=2.5, d=3.0, noise=0.5),
  "v3": dict(levels=7, alpha=6.0, cs=2.0, cx=2.0, d=2.5, noise=0.6),
}
N = 200; SEEDS = list(range(5)); GAMMAS = [1.0, 2.0, 3.0, 5.0, 8.0]
t0 = time.time()
for name, p in CONFIGS.items():
    mean, orc, eps = run(p, N, SEEDS, GAMMAS)
    # family bests = max over members over Γ
    XX = max(mean["IPW-X-X"].max(), mean["DR-X-X"].max())
    OX = max(mean["IPW-O-X"].max(), mean["DR-O-X"].max())
    OW = max(mean["IPW-O-W"].max(), mean["DR-O-W"].max())
    flag = "<<< OW>OX>XX" if (OW > OX + 0.005 and OX > XX + 0.005) else ""
    print("[%s] N=%d eps=%.4f,%.4f oracle=%.3f | XX=%.3f OX=%.3f OW=%.3f  (OW-OX=%+.3f OX-XX=%+.3f) %s"
          % (name, N, eps[0], eps[1], orc, XX, OX, OW, OW - OX, OX - XX, flag), flush=True)
    print("     DR by Γ: XX=%.3f  O-X=%s  O-W=%s" % (mean["DR-X-X"][0], np.round(mean["DR-O-X"], 3).tolist(), np.round(mean["DR-O-W"], 3).tolist()))
    print("     IPW byΓ: XX=%.3f  O-X=%s  O-W=%s" % (mean["IPW-X-X"][0], np.round(mean["IPW-O-X"], 3).tolist(), np.round(mean["IPW-O-W"], 3).tolist()))
print("done in %.1f min" % ((time.time() - t0) / 60))
