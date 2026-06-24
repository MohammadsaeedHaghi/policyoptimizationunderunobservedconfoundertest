"""Run all 8 methods on the E3 DGP, BOTH regimes (uncapped + capped), N=700, 5 seeds, Γ∈{1,2,4,8,20,100,1000}.
Exact analytic value. Saves owwin3_results.json: per-regime per-method 5-seed mean/sd by Γ + seed-0 policies + oracle."""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin3"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("ow3dgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["ow3dgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
S = {}
for reg, suf, Cap in [("uncap", "uncapped", "Uncapped"), ("cap", "capped", "Capped")]:
    S[(reg, "IPW-X-X")] = L("methods/IPW-X-X/%s/ipw_x_x_%s.py" % (Cap, suf), "solve_ipw_x_x_%s" % suf)
    S[(reg, "DoublyRobust-X-X")] = L("methods/DoublyRobust-X-X/%s/doublyrobust_x_x_%s.py" % (Cap, suf), "solve_doublyrobust_x_x_%s" % suf)
    S[(reg, "Direct-X-X")] = L("methods/Direct-X-X/%s/direct_x_x_%s.py" % (Cap, suf), "solve_direct_x_x_%s" % suf)
    S[(reg, "IPW-O-X")] = L("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf)
    S[(reg, "DoublyRobust-O-X")] = L("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf)
    S[(reg, "Hajek-O-X")] = L("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (Cap, suf), "solve_hajek_o_x_%s" % suf)
    S[(reg, "IPW-O-W")] = L("methods/IPW-O-W/%s/ipw_o_w_%s.py" % (Cap, suf), "solve_ipw_o_w_%s" % suf)
    S[(reg, "DoublyRobust-O-W")] = L("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf)
K = d.K; LV = d.LEVELS; NTR = 700; SEEDS = list(range(5)); GAMMAS = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
XXm = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]; OXm = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]; OWm = ["IPW-O-W", "DoublyRobust-O-W"]; ALL = XXm + OXm + OWm
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
def tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])
orc = d.exact_value(d.grid_truth()["oracle"])
out = {"N_train": NTR, "seeds": SEEDS, "gammas": GAMMAS, "oracle": round(orc, 4), "grid": [float(v) for v in LV], "regimes": {}}
t0 = time.time()
for reg in ["uncap", "cap"]:
    kw0 = {} if reg == "uncap" else {"cap": d.CAP}
    per = {m: [] for m in ALL}; polS = {m: {str(s): {} for s in SEEDS} for m in ALL}
    for sd in SEEDS:
        obs, _ = d.generate(NTR, sd); w, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K)
        wraw, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K, normalize=False)   # RAW (≥1) for Hajek-O
        mu = common.outcome_means(obs["X"], obs["T"], obs["Y"], n_arms=K, cross_fit=True)
        Dm = common.pairwise_distance_matrix(obs["X"]); eps = tuple(common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=1.0))
        flat = {"IPW-X-X": S[(reg, "IPW-X-X")](obs["X"], obs["T"], obs["Y"], w, n_arms=K, discretize=False, **kw0),
                "DoublyRobust-X-X": S[(reg, "DoublyRobust-X-X")](obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, discretize=False, **kw0),
                "Direct-X-X": S[(reg, "Direct-X-X")](obs["X"], obs["T"], obs["Y"], n_arms=K, discretize=False, **kw0)}
        for m, res in flat.items():
            per[m].append([d.exact_value(tg(res))] * len(GAMMAS))
            g0 = [round(float(v), 4) for v in tg(res)]
            for g in GAMMAS: polS[m][str(sd)][gk(g)] = g0
        for m in OXm + OWm: per[m].append([])
        for g in GAMMAS:
            res = {"IPW-O-X": S[(reg, "IPW-O-X")](obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, discretize=False, **kw0),
                   "DoublyRobust-O-X": S[(reg, "DoublyRobust-O-X")](obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, **kw0),
                   "Hajek-O-X": S[(reg, "Hajek-O-X")](obs["X"], obs["T"], obs["Y"], wraw, n_arms=K, Gamma=g, maximize=True, discretize=False, **kw0),
                   "IPW-O-W": S[(reg, "IPW-O-W")](obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0),
                   "DoublyRobust-O-W": S[(reg, "DoublyRobust-O-W")](obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0)}
            for m in OXm + OWm:
                per[m][-1].append(d.exact_value(tg(res[m])))
                polS[m][str(sd)][gk(g)] = [round(float(v), 4) for v in tg(res[m])]
        print("[%s] seed %d done (%.1f min)" % (reg, sd, (time.time() - t0) / 60), flush=True)
    # average policy across seeds (per method per Γ)
    for m in ALL:
        polS[m]["avg"] = {gk(g): [round(float(np.mean([polS[m][str(s)][gk(g)][j] for s in SEEDS])), 4) for j in range(len(LV))] for g in GAMMAS}
    out["regimes"][reg] = {"mean": {m: [round(float(v), 4) for v in np.array(per[m]).mean(0)] for m in ALL},
                           "sd": {m: [round(float(v), 4) for v in np.array(per[m]).std(0)] for m in ALL},
                           "policy_seed0": {m: polS[m]["0"] for m in ALL},
                           "policy_by_seed": polS}
(HERE / "owwin3_results.json").write_text(json.dumps(out, indent=2))
print("\nsaved owwin3_results.json  oracle=%.3f" % orc, flush=True)
for reg in ["uncap", "cap"]:
    M = out["regimes"][reg]["mean"]
    print("[%s] XX=%.3f OX(DR)=%s OW(DR)=%s" % (reg, M["DoublyRobust-X-X"][0], M["DoublyRobust-O-X"], M["DoublyRobust-O-W"]), flush=True)
