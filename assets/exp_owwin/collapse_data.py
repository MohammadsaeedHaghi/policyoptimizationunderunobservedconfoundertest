"""Large-Γ collapse data for the owwin tab: value + treat-fraction vs Γ (1..1000) for all 5 robust methods,
capped, N=700, 3 seeds, exact analytic value. Saves owwin_collapse.json."""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("owdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["owdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwox = L("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped")
drox  = L("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped")
hjox  = L("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")
ipwow = L("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped")
drow  = L("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped")
K = d.K; CAP = d.CAP; LV = d.LEVELS; NTR = 700; SEEDS = [0, 1, 2]
GAMMAS = [1.0, 2.0, 5.0, 8.0, 20.0, 50.0, 100.0, 300.0, 1000.0]
t = d.grid_truth(); never = float(np.mean(t["eY0"])); orc = d.exact_value(t["oracle"])
def tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])
METHODS = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
val = {m: {g: [] for g in GAMMAS} for m in METHODS}; frac = {m: {g: [] for g in GAMMAS} for m in METHODS}
t0 = time.time()
for sd in SEEDS:
    obs, _ = d.generate(NTR, sd); w, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K)
    mu = common.outcome_means(obs["X"], obs["T"], obs["Y"], n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(obs["X"]); eps = tuple(common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=1.0))
    for g in GAMMAS:
        res = {"IPW-O-X": ipwox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, cap=CAP, discretize=False),
               "DoublyRobust-O-X": drox(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, cap=CAP, discretize=False),
               "Hajek-O-X": hjox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, maximize=True, cap=CAP, discretize=False),
               "IPW-O-W": ipwow(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, cap=CAP, discretize=False, zscore=False, epsilon=eps),
               "DoublyRobust-O-W": drow(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, cap=CAP, discretize=False, zscore=False, epsilon=eps)}
        for m in METHODS:
            pg = tg(res[m]); val[m][g].append(d.exact_value(pg)); frac[m][g].append(float(pg.mean()))
    print("seed %d done (%.1f min)" % (sd, (time.time() - t0) / 60), flush=True)
out = {"N_train": NTR, "seeds": SEEDS, "gammas": GAMMAS, "never_treat": round(never, 4), "oracle": round(orc, 4),
       "value_mean": {m: [round(float(np.mean(val[m][g])), 4) for g in GAMMAS] for m in METHODS},
       "frac_mean": {m: [round(float(np.mean(frac[m][g])), 4) for g in GAMMAS] for m in METHODS}}
(HERE / "owwin_collapse.json").write_text(json.dumps(out, indent=2))
print("saved owwin_collapse.json (%.1f min)" % ((time.time() - t0) / 60), flush=True)
for m in METHODS: print("  %-16s value=%s" % (m, out["value_mean"][m]), flush=True)
