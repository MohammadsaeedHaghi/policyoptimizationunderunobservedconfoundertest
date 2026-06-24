"""CAPPED continuous-X 'Wasserstein wins' experiment.  Same DGP as run_wass.py but with a binding capacity cap
(treat <= 50%, the oracle's own treat fraction).  Hypothesis: with no cap the box-only IPW-O-X can hedge toward
treat-most and land a lucky Γ=4 peak; under the cap it must CHOOSE whom to treat, and a confounding-fooled box picks
the wrong half while the covariate-balancing Wasserstein method picks the right half -> IPW-O-W / DR-O-W win outright.
N_train=1000, one seed, Γ∈{1,2,4,8} (W skips Γ=1=capped parent X-X).  Saves wass_cap_results.json."""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_wass"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("wdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["wdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped")
drxx  = L("methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py", "solve_doublyrobust_x_x_capped")
ipwox = L("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped")
drox  = L("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped")
ipwow = L("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped")
drow  = L("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped")

K, N, MESH = 2, 1000, 11
CAP = (1.0, 0.5)                                  # control uncapped, treat <= 50%
G = d.GRID; GAMMAS = [1, 2, 4, 8]; SOLVE = [2, 4, 8]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
obs, full = d.generate(N, 0)
w, P = common.ipw_weights_from_data(obs["X"], obs["T"], K)        # logistic (mis-specified) propensity
muhat = common.outcome_means(obs["X"], obs["T"], obs["Y"], K)
_, tf = d.generate(40000, 99); Xt = tf["X"].ravel(); St = tf["S"]

def to_grid(res):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    nn = lv[np.argmin(np.abs(np.asarray(G)[:, None] - lv[None, :]), axis=1)]
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in nn])
def deploy(polG):
    nn = G[np.argmin(np.abs(Xt[:, None] - G[None, :]), axis=1)]
    pos = {round(float(G[i]), 6): i for i in range(len(G))}
    return np.array([polG[pos[round(float(v), 6)]] for v in nn])
def realised(res):
    pit = deploy(to_grid(res)); return float(np.mean(pit * d.mu1(Xt, St) + (1 - pit) * d.mu0(Xt, St)))
def tfrac(res): return float(np.mean(deploy(to_grid(res))))

# oracle under the cap = treat the top-50% by CATE; CATE=X so treat iff X>0 (exactly 50% mass) -> cap not violated
pol_or = d.oracle_policy(G); pit_or = deploy(pol_or)
oracle_val = float(np.mean(pit_or * d.mu1(Xt, St) + (1 - pit_or) * d.mu0(Xt, St)))
print("Oracle (treat iff X>0, frac=%.2f): %.4f" % (float(np.mean(pit_or)), oracle_val), flush=True)

# ε once, on the snapped geometry the W solver uses
sup = common.snap_to_grid(obs["X"], MESH, -1.0, 1.0)
Dm, _, _ = common.distance_matrix(sup, zscore=False)
eps = common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=1.0)
print("tight ε (per arm): %.5f , %.5f" % (eps[0], eps[1]), flush=True)

# Γ-free capped parents
r_xx = ipwxx(obs["X"], obs["T"], obs["Y"], w, n_arms=K, cap=CAP, discretize=True, mesh=MESH)
r_dx = drxx(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, cap=CAP, discretize=True, mesh=MESH)
v_xx, v_dx = realised(r_xx), realised(r_dx)
print("IPW-X-X=%.4f (frac %.2f)  DR-X-X=%.4f (frac %.2f)" % (v_xx, tfrac(r_xx), v_dx, tfrac(r_dx)), flush=True)

res = {"IPW-X-X": [round(v_xx, 4)] * len(GAMMAS), "DoublyRobust-X-X": [round(v_dx, 4)] * len(GAMMAS),
       "IPW-O-X": [], "DoublyRobust-O-X": [], "IPW-O-W": [], "DoublyRobust-O-W": []}
frac = {k: {} for k in res}; pol = {"IPW-O-X": {}, "DoublyRobust-O-X": {}, "IPW-O-W": {}, "DoublyRobust-O-W": {}}
for g in GAMMAS:
    t0 = time.time()
    rbo = ipwox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), cap=CAP, discretize=True, mesh=MESH)
    rbd = drox(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, Gamma=float(g), cap=CAP, discretize=True, mesh=MESH)
    res["IPW-O-X"].append(round(realised(rbo), 4)); res["DoublyRobust-O-X"].append(round(realised(rbd), 4))
    pol["IPW-O-X"][gk(g)] = [round(float(v), 4) for v in to_grid(rbo)]; pol["DoublyRobust-O-X"][gk(g)] = [round(float(v), 4) for v in to_grid(rbd)]
    if g == 1:
        res["IPW-O-W"].append(round(v_xx, 4)); res["DoublyRobust-O-W"].append(round(v_dx, 4))
        pol["IPW-O-W"][gk(g)] = [round(float(v), 4) for v in to_grid(r_xx)]; pol["DoublyRobust-O-W"][gk(g)] = [round(float(v), 4) for v in to_grid(r_dx)]
    else:
        rwo = ipwow(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), cap=CAP, discretize=True, mesh=MESH, zscore=False, epsilon=eps)
        rwd = drow(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, Gamma=float(g), cap=CAP, discretize=True, mesh=MESH, zscore=False, epsilon=eps)
        res["IPW-O-W"].append(round(realised(rwo), 4)); res["DoublyRobust-O-W"].append(round(realised(rwd), 4))
        pol["IPW-O-W"][gk(g)] = [round(float(v), 4) for v in to_grid(rwo)]; pol["DoublyRobust-O-W"][gk(g)] = [round(float(v), 4) for v in to_grid(rwd)]
    print("Γ=%d  O-X: IPW=%.4f DR=%.4f | O-W: IPW=%.4f DR=%.4f  (%.1f min)" %
          (g, res["IPW-O-X"][-1], res["DoublyRobust-O-X"][-1], res["IPW-O-W"][-1], res["DoublyRobust-O-W"][-1], (time.time() - t0) / 60), flush=True)

out = {"N_train": N, "N_test": 40000, "cap": list(CAP), "gammas": GAMMAS, "oracle": round(oracle_val, 4),
       "epsilon": [float(eps[0]), float(eps[1])], "value_by_gamma": res, "policy_by_gamma": pol, "grid": [float(v) for v in G]}
(HERE / "wass_cap_results.json").write_text(json.dumps(out, indent=2))
best_box = max(max(res["IPW-O-X"]), max(res["DoublyRobust-O-X"]))
best_w = max(max(res["IPW-O-W"]), max(res["DoublyRobust-O-W"]))
print("\nsaved wass_cap_results.json", flush=True)
print("BEST box (any Γ)=%.4f   BEST Wasserstein (any Γ)=%.4f   oracle=%.4f" % (best_box, best_w, oracle_val), flush=True)
print("IPW-O-X=%s IPW-O-W=%s" % (res["IPW-O-X"], res["IPW-O-W"]), flush=True)
print("DR-O-X=%s DR-O-W=%s" % (res["DoublyRobust-O-X"], res["DoublyRobust-O-W"]), flush=True)
print(">>> WASSERSTEIN WINS OUTRIGHT" if best_w > best_box + 1e-9 else ">>> box still >= Wasserstein", flush=True)
