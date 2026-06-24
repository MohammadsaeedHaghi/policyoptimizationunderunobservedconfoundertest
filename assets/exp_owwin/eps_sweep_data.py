"""ε-sweep for the OW methods (the user's 'try different epsilon'): uncapped owwin, c_eps in {0.5,1,2,4},
Γ swept to 1000, N=700, 3 seeds. Shows larger Wasserstein radius -> more conservative -> collapse at large Γ.
Saves owwin_epssweep.json (value + treat-frac per method per c_eps per Γ)."""
import sys, json, importlib.util, time, math
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("owdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["owdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwow = L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
drow  = L("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")
K = d.K; LV = d.LEVELS; N = 700; SEEDS = [0, 1, 2]; CEPS = [0.5, 1.0, 2.0, 4.0]; GAM = [2.0, 5.0, 8.0, 20.0, 50.0, 200.0, 1000.0]
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
def mval(fn, g, ce, dr):
    vs = []
    for obs, w, mu, Dm in prep:
        eps = tuple(common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=ce))
        r = fn(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps) if dr else fn(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps)
        pg = tg(r); vs.append((d.exact_value(pg), float(pg.mean())))
    return round(float(np.mean([v for v, _ in vs])), 4), round(float(np.mean([f for _, f in vs])), 3)
out = {"N_train": N, "seeds": SEEDS, "gammas": GAM, "c_eps": CEPS, "never_treat": round(never, 4), "oracle": round(orc, 4), "value": {}, "frac": {}}
t0 = time.time()
for nm, fn, dr in [("IPW-O-W", ipwow, False), ("DoublyRobust-O-W", drow, True)]:
    out["value"][nm] = {}; out["frac"][nm] = {}
    for ce in CEPS:
        vv = [mval(fn, g, ce, dr) for g in GAM]
        out["value"][nm]["%.1f" % ce] = [v for v, _ in vv]; out["frac"][nm]["%.1f" % ce] = [f for _, f in vv]
        print("%-16s c_eps=%.1f val=%s frac=%s (%.1f min)" % (nm, ce, [v for v, _ in vv], [f for _, f in vv], (time.time() - t0) / 60), flush=True)
(HERE / "owwin_epssweep.json").write_text(json.dumps(out, indent=2))
print("saved owwin_epssweep.json", flush=True)
