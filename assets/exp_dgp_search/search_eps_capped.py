"""CAPPED owwin: find an ε (c_eps) that gives BOTH (1) XX<OX<OW at moderate Γ and (2) OW collapses at large Γ.
Sweep c_eps in {1,2,4} over full Γ grid; report XX/OX baselines + OW value & treat-frac. N=500, 3 seeds, exact value."""
import sys, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(ROOT)); import common
HERE = ROOT / "assets" / "exp_owwin"
sp = importlib.util.spec_from_file_location("owdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["owdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped")
drxx  = L("methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py", "solve_doublyrobust_x_x_capped")
ipwox = L("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped")
drox  = L("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped")
ipwow = L("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped")
drow  = L("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped")
K = d.K; LV = d.LEVELS; CAP = d.CAP; N = 500; SEEDS = [0, 1, 2]; GAM = [2.0, 5.0, 8.0, 20.0, 50.0, 200.0, 1000.0]
t = d.grid_truth(); never = float(np.mean(t["eY0"])); orc = d.exact_value(t["oracle"])
def tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])
prep = []
for sd in SEEDS:
    obs, _ = d.generate(N, sd); w, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K)
    mu = common.outcome_means(obs["X"], obs["T"], obs["Y"], n_arms=K, cross_fit=True); Dm = common.pairwise_distance_matrix(obs["X"])
    prep.append((obs, w, mu, Dm))
def mval(fn, g=None, ceps=None, dr=False, ow=False):
    vs = []
    for obs, w, mu, Dm in prep:
        if ow:
            eps = tuple(common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=ceps))
            r = fn(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, cap=CAP, discretize=False, zscore=False, epsilon=eps) if dr else fn(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, cap=CAP, discretize=False, zscore=False, epsilon=eps)
        elif g is not None:
            r = fn(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, cap=CAP, discretize=False) if dr else fn(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, cap=CAP, discretize=False)
        else:
            r = fn(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, cap=CAP, discretize=False) if dr else fn(obs["X"], obs["T"], obs["Y"], w, cap=CAP, n_arms=K, discretize=False)
        vs.append((d.exact_value(tg(r)), float(tg(r).mean())))
    return float(np.mean([v for v, _ in vs])), float(np.mean([f for _, f in vs]))
t0 = time.time()
print("never-treat=%.3f oracle=%.3f  | XX: IPW=%.3f DR=%.3f" % (never, orc, mval(ipwxx)[0], mval(drxx, dr=True)[0]), flush=True)
print("Γ grid:", GAM, flush=True)
for name, fn, dr in [("IPW-O-X", ipwox, False), ("DR-O-X", drox, True)]:
    print("  %-8s = %s" % (name, [round(mval(fn, g=g, dr=dr)[0], 3) for g in GAM]), flush=True)
for ce in [1.0, 2.0, 4.0]:
    print("--- c_eps=%.1f ---" % ce, flush=True)
    for name, fn, dr in [("IPW-O-W", ipwow, False), ("DR-O-W", drow, True)]:
        vv = [mval(fn, g=g, ceps=ce, dr=dr, ow=True) for g in GAM]
        print("  %-8s val=%s  frac=%s" % (name, [round(v, 3) for v, _ in vv], [round(f, 2) for _, f in vv]), flush=True)
print("done %.1f min" % ((time.time() - t0) / 60), flush=True)
