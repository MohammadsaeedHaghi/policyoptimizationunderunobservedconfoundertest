"""Discrete-X version of the 'Wasserstein wins' experiment.  N_train=1000, one seed, Γ∈{1,2,4,8}.
Runs the full family: IPW-{X-X,O-X,O-W} and DoublyRobust-{X-X,O-X,O-W}, plus the Oracle ceiling.
Reports the tight Wasserstein radius ε under BOTH propensities:
  - logistic (mis-specified, parametric)  -> ε>0 even on discrete X  -> Wasserstein has something to correct
  - counting (exact per-cell Hajek)        -> ε=0 on discrete X       -> Wasserstein is moot (the HR regime)
Evaluates every policy on a 40k test set with the true means.  Saves disc_results.json.
"""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_wass_disc"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("ddgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["ddgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped")
drxx  = L("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped")
ipwox = L("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
drox  = L("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped")
ipwow = L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
drow  = L("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")

K, N = 2, 1000
G = d.GRID; GAMMAS = [1, 2, 4, 8]; SOLVE = [2, 4, 8]
obs, full = d.generate(N, 0)
w, P = common.ipw_weights_from_data(obs["X"], obs["T"], K)          # LOGISTIC (mis-specified) propensity
muhat = common.outcome_means(obs["X"], obs["T"], obs["Y"], K)
_, tf = d.generate(40000, 99); Xt = tf["X"].ravel(); St = tf["S"]

# ---- ε contrast: logistic vs counting, on the raw discrete support ----
Xcol = np.asarray(obs["X"], float).reshape(-1, 1)
Dm, _, _ = common.distance_matrix(Xcol, zscore=False)
eps_log = common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=1.0)
wc, _ = common.count_ipw_weights_from_data(obs["X"], obs["T"], K)   # COUNTING (exact per-cell) propensity
eps_cnt = common.tight_epsilon(Dm, obs["T"], wc, K, is_distance=True, c_eps=1.0)
print("tight ε  (logistic) : %.5f , %.5f" % (eps_log[0], eps_log[1]), flush=True)
print("tight ε  (counting) : %.5f , %.5f" % (eps_cnt[0], eps_cnt[1]), flush=True)
eps = eps_log                                                       # the W methods use the logistic ε

def to_grid(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    nn = lv[np.argmin(np.abs(np.asarray(G)[:, None] - lv[None, :]), axis=1)]
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in nn])
def realised(res):
    pol = to_grid(res)
    nn = G[np.argmin(np.abs(Xt[:, None] - G[None, :]), axis=1)]
    pos = {round(float(G[i]), 6): i for i in range(len(G))}
    pit = np.array([pol[pos[round(float(v), 6)]] for v in nn])
    return float(np.mean(pit * d.mu1(Xt, St) + (1 - pit) * d.mu0(Xt, St)))

# ---- oracle ceiling (treat iff X>0, with true means) ----
gt = d.grid_truth(); pol_or = gt["oracle"]
nn = G[np.argmin(np.abs(Xt[:, None] - G[None, :]), axis=1)]
pos = {round(float(G[i]), 6): i for i in range(len(G))}
pit_or = np.array([pol_or[pos[round(float(v), 6)]] for v in nn])
oracle_val = float(np.mean(pit_or * d.mu1(Xt, St) + (1 - pit_or) * d.mu0(Xt, St)))
print("Oracle (treat iff X>0): %.4f" % oracle_val, flush=True)

# ---- Γ=1 anchors (parent X-X), used for the O-W series at Γ=1 ----
r_ipwxx = ipwxx(obs["X"], obs["T"], obs["Y"], w, n_arms=K, discretize=False)
r_drxx  = drxx(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, discretize=False)
v_ipwxx, v_drxx = realised(r_ipwxx), realised(r_drxx)
print("IPW-X-X=%.4f  DoublyRobust-X-X=%.4f  (Γ-free)" % (v_ipwxx, v_drxx), flush=True)

res = {"IPW-X-X": [v_ipwxx] * len(GAMMAS), "DoublyRobust-X-X": [v_drxx] * len(GAMMAS),
       "IPW-O-X": [], "DoublyRobust-O-X": [], "IPW-O-W": [], "DoublyRobust-O-W": []}
pol = {"IPW-O-X": {}, "DoublyRobust-O-X": {}, "IPW-O-W": {}, "DoublyRobust-O-W": {}}
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)

for g in GAMMAS:
    t0 = time.time()
    rbo = ipwox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), discretize=False)
    rbd = drox(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, Gamma=float(g), discretize=False)
    res["IPW-O-X"].append(round(realised(rbo), 4)); res["DoublyRobust-O-X"].append(round(realised(rbd), 4))
    pol["IPW-O-X"][gk(g)] = [round(float(v), 4) for v in to_grid(rbo)]
    pol["DoublyRobust-O-X"][gk(g)] = [round(float(v), 4) for v in to_grid(rbd)]
    if g == 1:                                                      # W@Γ1 = parent X-X (provably)
        res["IPW-O-W"].append(round(v_ipwxx, 4)); res["DoublyRobust-O-W"].append(round(v_drxx, 4))
        pol["IPW-O-W"][gk(g)] = [round(float(v), 4) for v in to_grid(r_ipwxx)]
        pol["DoublyRobust-O-W"][gk(g)] = [round(float(v), 4) for v in to_grid(r_drxx)]
    else:
        rwo = ipwow(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), discretize=False, zscore=False, epsilon=eps)
        rwd = drow(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, Gamma=float(g), discretize=False, zscore=False, epsilon=eps)
        res["IPW-O-W"].append(round(realised(rwo), 4)); res["DoublyRobust-O-W"].append(round(realised(rwd), 4))
        pol["IPW-O-W"][gk(g)] = [round(float(v), 4) for v in to_grid(rwo)]
        pol["DoublyRobust-O-W"][gk(g)] = [round(float(v), 4) for v in to_grid(rwd)]
    print("Γ=%d  O-X: IPW=%.4f DR=%.4f | O-W: IPW=%.4f DR=%.4f  (%.1f min)" %
          (g, res["IPW-O-X"][-1], res["DoublyRobust-O-X"][-1], res["IPW-O-W"][-1], res["DoublyRobust-O-W"][-1], (time.time() - t0) / 60), flush=True)

out = {"N_train": N, "N_test": 40000, "gammas": GAMMAS, "oracle": round(oracle_val, 4),
       "epsilon_logistic": [float(eps_log[0]), float(eps_log[1])], "epsilon_counting": [float(eps_cnt[0]), float(eps_cnt[1])],
       "value_by_gamma": res, "policy_by_gamma": pol, "grid": [float(v) for v in G]}
(HERE / "disc_results.json").write_text(json.dumps(out, indent=2))
print("\nsaved disc_results.json", flush=True)
print("Oracle=%.4f  IPW-O-X byΓ=%s  IPW-O-W byΓ=%s" % (oracle_val, res["IPW-O-X"], res["IPW-O-W"]), flush=True)
print("DR-O-X byΓ=%s  DR-O-W byΓ=%s" % (res["DoublyRobust-O-X"], res["DoublyRobust-O-W"]), flush=True)
