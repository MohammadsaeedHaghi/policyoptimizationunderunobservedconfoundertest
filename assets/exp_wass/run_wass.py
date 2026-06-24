"""Run the Wasserstein methods (IPW-O-W, DoublyRobust-O-W) on the continuous-X experiment at N_train=1000,
over Γ ∈ {2,4,8} (Γ=1 is NOT solved — it equals the parent X-X method: IPW-O-W@Γ1=IPW-X-X, DR-O-W@Γ1=DR-X-X).
Evaluate on the 40k test set; patch WASS_CHARTS (value-vs-Γ + policy) in index.html. ε precomputed once. ~12 min/solve."""
import sys, json, re, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_wass"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("wdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["wdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwow = L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
drow = L("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")
ipwxx = L("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped")
drxx = L("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped")

K, N, MESH = 2, 1000, 11
G = d.GRID; GAMMAS = [1, 2, 4, 8]; SOLVE = [2, 4, 8]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
obs, full = d.generate(N, 0)
w, P = common.ipw_weights_from_data(obs["X"], obs["T"], K)
muhat = common.outcome_means(obs["X"], obs["T"], obs["Y"], K)
_, tf = d.generate(40000, 99); Xt = tf["X"].ravel(); St = tf["S"]

def to_grid(res, Xq):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    nn = lv[np.argmin(np.abs(np.asarray(Xq)[:, None] - lv[None, :]), axis=1)]
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in nn])
def realised(res): p = to_grid(res, Xt); return float(np.mean(p * d.mu1(Xt, St) + (1 - p) * d.mu0(Xt, St)))

# Γ=1 anchors: IPW-O-W = IPW-X-X, DR-O-W = DR-X-X (NOT solved — provably equal)
rxx = ipwxx(obs["X"], obs["T"], obs["Y"], w, n_arms=K, discretize=True, mesh=MESH)
rdx = drxx(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, discretize=True, mesh=MESH)
anchor_ow = (realised(rxx), [round(float(v), 4) for v in to_grid(rxx, G)])
anchor_dw = (realised(rdx), [round(float(v), 4) for v in to_grid(rdx, G)])
print("Γ=1 (copied): IPW-O-W=IPW-X-X=%.4f  DR-O-W=DR-X-X=%.4f" % (anchor_ow[0], anchor_dw[0]), flush=True)

# ε once, on the SAME snapped geometry the solver uses
sup = common.snap_to_grid(obs["X"], MESH, -1.0, 1.0)
D, _, _ = common.distance_matrix(sup, zscore=False)
eps = common.tight_epsilon(D, obs["T"], w, K, is_distance=True, c_eps=1.0)
print("tight ε (per arm): %.5f , %.5f" % (eps[0], eps[1]), flush=True)

ow, dw = {}, {}
for g in SOLVE:
    t0 = time.time()
    ow[g] = ipwow(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), discretize=True, mesh=MESH, zscore=False, epsilon=eps)
    dw[g] = drow(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, Gamma=float(g), discretize=True, mesh=MESH, zscore=False, epsilon=eps)
    print("Γ=%d  IPW-O-W=%.4f  DR-O-W=%.4f  (%.1f min)" % (g, realised(ow[g]), realised(dw[g]), (time.time() - t0) / 60), flush=True)

vO = [anchor_ow[0] if g == 1 else round(realised(ow[g]), 4) for g in GAMMAS]
vD = [anchor_dw[0] if g == 1 else round(realised(dw[g]), 4) for g in GAMMAS]
polO = {gk(g): (anchor_ow[1] if g == 1 else [round(float(v), 4) for v in to_grid(ow[g], G)]) for g in GAMMAS}
polD = {gk(g): (anchor_dw[1] if g == 1 else [round(float(v), 4) for v in to_grid(dw[g], G)]) for g in GAMMAS}
(HERE / "wass_wasserstein.json").write_text(json.dumps({"epsilon": list(eps), "gammas": GAMMAS, "IPW-O-W": vO, "DoublyRobust-O-W": vD,
    "policy_IPW-O-W": polO, "policy_DR-O-W": polD}, indent=2))
print("IPW-O-W byΓ:", vO, " DR-O-W byΓ:", vD, flush=True)

# ---- patch WASS_CHARTS ----
idx = ROOT / "index.html"; h = idx.read_text()
m = re.search(r'var WASS_CHARTS = (.*?);\n', h); W = json.loads(m.group(1))
def ser(idl, color, y): return {"id": idl, "label": idl, "color": color, "y": y}
W["value"]["series"] = [s for s in W["value"]["series"] if s["id"] not in ("IPW-O-W", "DoublyRobust-O-W")]
W["value"]["series"].insert(-1, ser("IPW-O-W", "#1f77b4", vO))
W["value"]["series"].insert(-1, ser("DoublyRobust-O-W", "#2ca02c", vD))
for g in GAMMAS:
    key = gk(g); s = W["policy"]["seriesByGamma"][key]
    s[:] = [x for x in s if x["id"] not in ("IPW-O-W", "DoublyRobust-O-W")]
    s.insert(-1, ser("IPW-O-W", "#1f77b4", polO[key]))
    s.insert(-1, ser("DoublyRobust-O-W", "#2ca02c", polD[key]))
W["value"]["note"] = "Realised test E[Y] vs Γ, N_train=1000. Γ∈{1,2,4,8}; W methods at Γ=1 = parent X-X (not re-solved). Exact per-unit transport."
W["policy"]["note"] = "Deployed π(treat|X), N_train=1000. Γ-free baselines + box-robust + Wasserstein. Γ dropdown; W@Γ=1 = X-X."
h = h[:m.start()] + "var WASS_CHARTS = " + json.dumps(W, separators=(",", ":")) + ";\n" + h[m.end():]
idx.write_text(h)
print("patched WASS_CHARTS value+policy", flush=True)
