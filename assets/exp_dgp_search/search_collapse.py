"""Search for a DGP with BOTH: (1) XX<OX<OW at moderate Γ, and (2) all methods degrade at large Γ (collapse).
Hypothesis: the capacity cap blocks the value methods from hedging to never-treat. Test UNCAPPED vs CAPPED on the
owwin DGP, sweep Γ to 1000. Also sweep the OW Wasserstein radius (c_eps). N=500, 3 seeds, exact analytic value."""
import sys, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(ROOT)); import common
HERE = ROOT / "assets" / "exp_owwin"
sp = importlib.util.spec_from_file_location("owdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["owdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
S = {
 ("uncap", "IPW-X-X"): L("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped"),
 ("uncap", "DR-X-X"): L("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped"),
 ("uncap", "IPW-O-X"): L("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped"),
 ("uncap", "DR-O-X"): L("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped"),
 ("uncap", "IPW-O-W"): L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped"),
 ("uncap", "DR-O-W"): L("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped"),
 ("cap", "IPW-X-X"): L("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped"),
 ("cap", "DR-X-X"): L("methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py", "solve_doublyrobust_x_x_capped"),
 ("cap", "IPW-O-X"): L("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped"),
 ("cap", "DR-O-X"): L("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped"),
 ("cap", "IPW-O-W"): L("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped"),
 ("cap", "DR-O-W"): L("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped"),
}
K = d.K; LV = d.LEVELS; N = 500; SEEDS = [0, 1, 2]; CAPV = {"uncap": (1.0, 1.0), "cap": (1.0, 0.5)}
GAM = [2.0, 5.0, 8.0, 20.0, 100.0, 1000.0]
t = d.grid_truth(); never = float(np.mean(t["eY0"])); orc = d.exact_value(t["oracle"])
def tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])
print("never-treat=%.3f  oracle=%.3f" % (never, orc), flush=True)
data = [(sd, d.generate(N, sd)[0]) for sd in SEEDS]
prep = {}
for sd, obs in data:
    w, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K)
    mu = common.outcome_means(obs["X"], obs["T"], obs["Y"], n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(obs["X"])
    prep[sd] = (obs, w, mu, Dm)
def val(reg, name, g, ceps=1.0):
    vs = []
    for sd, obs in data:
        _, w, mu, Dm = prep[sd]; kw = {} if reg == "uncap" else {"cap": CAPV[reg]}
        f = S[(reg, name)]
        if name.endswith("O-W"):
            eps = tuple(common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=ceps))
            r = f(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw) if name.startswith("DR") else f(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw)
        elif name.endswith("O-X"):
            r = f(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, **kw) if name.startswith("DR") else f(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, discretize=False, **kw)
        else:
            r = f(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, discretize=False, **kw) if name.startswith("DR") else f(obs["X"], obs["T"], obs["Y"], w, n_arms=K, discretize=False, **kw)
        vs.append(d.exact_value(tg(r)))
    return float(np.mean(vs))
t0 = time.time()
for reg in ("uncap", "cap"):
    print("\n=== %s (cap=%s) ===" % (reg, CAPV[reg]), flush=True)
    xx = max(val(reg, "IPW-X-X", 1), val(reg, "DR-X-X", 1))
    print("  XX (Γ-free) best = %.3f" % xx, flush=True)
    for name in ["IPW-O-X", "DR-O-X", "IPW-O-W", "DR-O-W"]:
        row = [val(reg, name, g) for g in GAM]
        print("  %-8s byΓ%s = %s" % (name, GAM, np.round(row, 3).tolist()), flush=True)
print("\n=== ε sweep (uncap, DR-O-W) c_eps in {0.5,1,2,4} at Γ=8 and Γ=100 ===", flush=True)
for ce in [0.5, 1.0, 2.0, 4.0]:
    print("  c_eps=%.1f : Γ=8 -> %.3f   Γ=100 -> %.3f" % (ce, val("uncap", "DR-O-W", 8.0, ce), val("uncap", "DR-O-W", 100.0, ce)), flush=True)
print("done %.1f min" % ((time.time() - t0) / 60), flush=True)
