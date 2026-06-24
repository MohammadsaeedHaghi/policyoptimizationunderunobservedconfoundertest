"""Run all 8 methods on the exp_owwin DGP, capped (treat<=50%), N_train=700, 8 seeds, Γ∈{1,2,3,5,8,12}.
Exact analytic value (X uniform on grid). Saves owwin_results.json: per-method 8-seed mean/sd value by Γ,
seed-0 deployed policy grids by Γ (for the π chart), and the oracle. ~40 min (OW transport solves)."""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("owdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["owdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped")
drxx  = L("methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py", "solve_doublyrobust_x_x_capped")
dirxx = L("methods/Direct-X-X/Capped/direct_x_x_capped.py", "solve_direct_x_x_capped")
ipwox = L("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped")
drox  = L("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped")
hjox  = L("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")
ipwow = L("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped")
drow  = L("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped")

K = d.K; CAP = d.CAP; LV = d.LEVELS; NTR = 700; SEEDS = list(range(8)); GAMMAS = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
XXm = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]; OXm = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]; OWm = ["IPW-O-W", "DoublyRobust-O-W"]
ALL = XXm + OXm + OWm
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
def tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])

per = {m: [] for m in ALL}; pol0 = {m: {} for m in ALL}
orc = d.exact_value(d.grid_truth()["oracle"])
t0 = time.time()
for sd in SEEDS:
    obs, _ = d.generate(NTR, sd)
    w, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K)
    mu = common.outcome_means(obs["X"], obs["T"], obs["Y"], n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(obs["X"]); eps = tuple(common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=1.0))
    rx = ipwxx(obs["X"], obs["T"], obs["Y"], w, n_arms=K, cap=CAP, discretize=False)
    rd = drxx(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, cap=CAP, discretize=False)
    rr = dirxx(obs["X"], obs["T"], obs["Y"], n_arms=K, cap=CAP, discretize=False)
    flat = {"IPW-X-X": rx, "DoublyRobust-X-X": rd, "Direct-X-X": rr}
    for m, res in flat.items():
        per[m].append([d.exact_value(tg(res))] * len(GAMMAS))
        if sd == 0:
            g0 = [round(float(v), 4) for v in tg(res)]
            for g in GAMMAS: pol0[m][gk(g)] = g0
    for m in OXm + OWm: per[m].append([])
    for g in GAMMAS:
        res = {"IPW-O-X": ipwox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, cap=CAP, discretize=False),
               "DoublyRobust-O-X": drox(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, cap=CAP, discretize=False),
               "Hajek-O-X": hjox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, maximize=True, cap=CAP, discretize=False),
               "IPW-O-W": ipwow(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, cap=CAP, discretize=False, zscore=False, epsilon=eps),
               "DoublyRobust-O-W": drow(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, cap=CAP, discretize=False, zscore=False, epsilon=eps)}
        for m in OXm + OWm:
            per[m][-1].append(d.exact_value(tg(res[m])))
            if sd == 0: pol0[m][gk(g)] = [round(float(v), 4) for v in tg(res[m])]
    print("seed %d done (%.1f min)" % (sd, (time.time() - t0) / 60), flush=True)

mean = {m: [round(float(v), 4) for v in np.array(per[m]).mean(0)] for m in ALL}
sd_ = {m: [round(float(v), 4) for v in np.array(per[m]).std(0)] for m in ALL}
out = {"N_train": NTR, "seeds": SEEDS, "gammas": GAMMAS, "oracle": round(orc, 4), "cap": list(CAP),
       "epsilon_last": list(eps), "mean": mean, "sd": sd_, "policy_seed0": pol0,
       "grid": [float(v) for v in LV]}
(HERE / "owwin_results.json").write_text(json.dumps(out, indent=2))
print("\nsaved owwin_results.json  oracle=%.3f" % orc, flush=True)
for fam, ms in (("XX", XXm), ("OX", OXm), ("OW", OWm)):
    print(fam, {m: mean[m] for m in ms}, flush=True)
