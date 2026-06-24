"""Confirm the large-Γ collapse on the owwin DGP: run the OX (and a few OW) methods at increasingly extreme Γ
and report realised value + treat-fraction. Collapse = treat-fraction -> 0 (never-treat) and value -> never-treat
baseline. Capped, N=700, 3 seeds, exact analytic value. Box methods at all Γ; OW at a subset (transport is slow)."""
import sys, importlib.util
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
t = d.grid_truth()
never = float(np.mean(t["eY0"])); allt = float(np.mean(t["eY1"])); orc = d.exact_value(t["oracle"])
print("baselines: never-treat=%.3f  treat-all=%.3f  oracle=%.3f  (cap<=50%%)" % (never, allt, orc), flush=True)
def tg(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])
GBOX = [8, 20, 50, 100, 300, 1000]; GOW = [8, 50, 1000]
def runseeds(solver_call, gammas):
    out = {g: [] for g in gammas}
    for sd in SEEDS:
        obs, _ = d.generate(NTR, sd); w, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K)
        mu = common.outcome_means(obs["X"], obs["T"], obs["Y"], n_arms=K, cross_fit=True)
        eps = None
        for g in gammas:
            if eps is None and solver_call.__name__.endswith("w_capped"):
                Dm = common.pairwise_distance_matrix(obs["X"]); eps = tuple(common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=1.0))
            pg = tg(solver_call(obs, w, mu, g, eps)); out[g].append((d.exact_value(pg), float(pg.mean())))
    return {g: (float(np.mean([v for v, _ in out[g]])), float(np.mean([f for _, f in out[g]]))) for g in gammas}

calls = {
 "IPW-O-X": lambda obs, w, mu, g, eps: ipwox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), cap=CAP, discretize=False),
 "DoublyRobust-O-X": lambda obs, w, mu, g, eps: drox(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=float(g), cap=CAP, discretize=False),
 "Hajek-O-X": lambda obs, w, mu, g, eps: hjox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), maximize=True, cap=CAP, discretize=False),
 "IPW-O-W": lambda obs, w, mu, g, eps: ipwow(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), cap=CAP, discretize=False, zscore=False, epsilon=eps),
 "DoublyRobust-O-W": lambda obs, w, mu, g, eps: drow(obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=float(g), cap=CAP, discretize=False, zscore=False, epsilon=eps),
}
print("\n--- OX family (value | treat-frac), 3-seed mean ---", flush=True)
for nm in ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]:
    r = runseeds(calls[nm], GBOX)
    print("%-16s %s" % (nm, "  ".join("Γ=%-4g:%.3f|%.2f" % (g, r[g][0], r[g][1]) for g in GBOX)), flush=True)
print("\n--- OW family (value | treat-frac), 3-seed mean ---", flush=True)
for nm in ["IPW-O-W", "DoublyRobust-O-W"]:
    r = runseeds(calls[nm], GOW)
    print("%-16s %s" % (nm, "  ".join("Γ=%-4g:%.3f|%.2f" % (g, r[g][0], r[g][1]) for g in GOW)), flush=True)
print("\n(collapse = treat-frac->0 and value->never-treat %.3f)" % never, flush=True)
