"""Run the 'second experiment' (2-arm, 2-D discrete-X) for ONE seed, in BOTH regimes (UNCAPPED cap=(1,1) and
CAPPED cap=(1,0.5)). 9 methods (R-OW, R-O, R-OW-DR, R-O-DR, IPW, AIPW, Regret-O, Hajek-OW, Kallus) × Γ sweep.
Wasserstein ground cost uses zscore=True (2-D). Records realised train/test E[Y], worst-case objective, treat-fraction;
+ capacity-constrained ceilings. Saves per-(method,Γ) treat-prob grids over the 11×11=121 grid for ALL seeds (for the
interactive heatmap). Saves _multiseed/seed{seed}.json.  Usage: python3 run_multiseed.py {seed}"""
import sys, json, importlib.util
import numpy as np
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
SEED = int(sys.argv[1])
NEW = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(NEW)); import common
HERE = NEW / "assets" / "exp_second"
sp = importlib.util.spec_from_file_location("dgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
def load(n, rel):
    s = importlib.util.spec_from_file_location(n, str(NEW / rel)); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
LPf = lambda rel, fn: getattr(load(fn, rel), fn)
row = LPf("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped")
ro = LPf("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped")
ipw = LPf("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped")
rowdr = LPf("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped")
rodr = LPf("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped")
rego = LPf("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")
regow = LPf("methods/Hajek-O-W/Capped/hajek_o_w_capped.py", "solve_hajek_o_w_capped")
kal = load("kal", "methods/Kallus/kallus.py")
K = dgp.K; GAMMAS = dgp.GAMMAS; mi = dgp.mi; MG = GAMMAS[mi]; ZS = True   # 2-D ground cost -> zscore ON
GVAR = ["R-OW", "R-O", "R-OW-DR", "R-O-DR", "Regret-O", "Hajek-OW"]; METHODS = GVAR + ["IPW", "AIPW", "Kallus"]
REGIMES = [("uncap", dgp.CAP_UNCAP), ("cap", dgp.CAP_CAP)]

def constrained_oracle(X, V, cap):
    """max Σ_i Σ_k π_k(x_i)V[i,k] s.t. per-unit simplex, same-X tying (unique 2-D rows), (1/n)Σ π_k ≤ cap_k. Returns (K,n)."""
    Xr = np.round(np.asarray(X, float), 6); uniq, inv = np.unique(Xr, axis=0, return_inverse=True); n = len(Xr); ns = len(uniq)
    Vs = np.zeros((ns, K)); cnt = np.zeros(ns)
    for u in range(ns):
        m = inv == u; cnt[u] = m.sum()
        for k in range(K): Vs[u, k] = V[m, k].sum()
    mdl = gp.Model(); mdl.Params.OutputFlag = 0; pi = mdl.addVars(ns, K, lb=0, ub=1)
    mdl.setObjective(gp.quicksum(pi[s, k] * Vs[s, k] for s in range(ns) for k in range(K)), GRB.MAXIMIZE)
    for s in range(ns): mdl.addConstr(gp.quicksum(pi[s, k] for k in range(K)) == 1)
    for k in range(K): mdl.addConstr(gp.quicksum(cnt[s] * pi[s, k] for s in range(ns)) <= cap[k] * n)
    mdl.optimize(); out = np.zeros((K, n))
    for u in range(ns):
        m = inv == u
        for k in range(K): out[k, m] = pi[u, k].X
    return out

rng = np.random.default_rng(SEED); tr = dgp.generate(dgp.n_tr, rng); te = dgp.generate(dgp.n_te, rng)
w, _ = common.ipw_weights_from_data(tr.X, tr.T, K)
muhat = common.outcome_means(tr.X, tr.T, tr.Y, n_arms=K, cross_fit=True)
# Wasserstein methods compute their tight ε internally from their OWN z-scored ground cost (c_eps=1.0) — do NOT
# override epsilon here: a raw-X ε would mismatch the z-scored D and make the inner LP unbounded.
# deploy: nearest train unit (exact when the grid cell is present in train)
GRIDcol = np.argmin(((dgp.GRID[:, None, :] - tr.X[None, :, :]) ** 2).sum(2), axis=1)   # (121,)
nn_te = np.argmin(((te.X[:, None, :] - tr.X[None, :, :]) ** 2).sum(2), axis=1)          # (n_te,)
def metrics(pi):
    pi = np.asarray(pi, float)
    rt_tr = float((pi * tr.Ypot.T).sum() / pi.shape[1]); pte = pi[:, nn_te]
    rt_te = float((pte * te.Ypot.T).sum() / pte.shape[1]); treat = float(pi[1].mean())
    return rt_tr, rt_te, treat
def kal_metrics(pi_tr, pi_te):
    return float((pi_tr * tr.Ypot).sum(1).mean()), float((pi_te * te.Ypot).sum(1).mean()), float(pi_tr[:, 1].mean())

out = {"seed": SEED, "gammas": GAMMAS, "gamma_conf": dgp.GAMMA_CONF, "matched_gamma": MG, "regimes": {}}
for rname, cap in REGIMES:
    R = {"cap": list(cap), "methods": {m: {} for m in METHODS}}
    fi = constrained_oracle(te.X, te.Ypot, cap); bm = constrained_oracle(te.X, te.mu, cap)
    fitr = constrained_oracle(tr.X, tr.Ypot, cap); bmtr = constrained_oracle(tr.X, tr.mu, cap)
    R["ceilings"] = {"full_info_test": float((fi * te.Ypot.T).sum() / te.Ypot.shape[0]),
                     "best_means_test": float((bm * te.mu.T).sum() / te.mu.shape[0]),
                     "full_info_train": float((fitr * tr.Ypot.T).sum() / tr.Ypot.shape[0]),
                     "best_means_train": float((bmtr * tr.mu.T).sum() / tr.mu.shape[0])}
    pi_ipw = ipw(tr.X, tr.T, tr.Y, w, n_arms=K, cap=cap, discretize=False).pi; m_ipw = metrics(pi_ipw)
    pi_ai = rodr(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=1.0, cap=cap, discretize=False).pi; m_ai = metrics(pi_ai)
    grids = {}
    for gi, G in enumerate(GAMMAS):
        R["methods"]["IPW"][gi] = {"rt_train": m_ipw[0], "rt_test": m_ipw[1], "treat": m_ipw[2], "obj": float("nan")}
        R["methods"]["AIPW"][gi] = {"rt_train": m_ai[0], "rt_test": m_ai[1], "treat": m_ai[2], "obj": float("nan")}
        sv = {}
        sv["R-OW"] = row(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, cap=cap, discretize=False, zscore=ZS)
        sv["R-O"] = ro(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, cap=cap, discretize=False)
        sv["R-OW-DR"] = rowdr(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=G, cap=cap, discretize=False, zscore=ZS)
        sv["R-O-DR"] = rodr(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=G, cap=cap, discretize=False)
        sv["Regret-O"] = rego(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, maximize=True, cap=cap, discretize=False)
        sv["Hajek-OW"] = regow(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, maximize=True, cap=cap, discretize=False, zscore=ZS)
        for m in GVAR:
            mm = metrics(sv[m].pi)
            R["methods"][m][gi] = {"rt_train": mm[0], "rt_test": mm[1], "treat": mm[2], "obj": float(getattr(sv[m], "objective_value", float("nan")))}
        try:
            th = kal.fit_kallus(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(G), maximize=True, wasserstein=False, basis=("affine",), n_iters=20, n_restarts=4, seed=0)
            pit = kal.predict_kallus(th.theta, tr.X, ("affine",)); pie = kal.predict_kallus(th.theta, te.X, ("affine",))
            km = kal_metrics(pit, pie)
            R["methods"]["Kallus"][gi] = {"rt_train": km[0], "rt_test": km[1], "treat": km[2], "obj": float(th.objective_value)}
        except Exception:
            R["methods"]["Kallus"][gi] = {"rt_train": None, "rt_test": None, "treat": None, "obj": None}
        # treat-prob grid over the 121 grid cells (row 1 = treat)
        g = {m: [round(float(v), 4) for v in np.asarray(sv[m].pi, float)[:, GRIDcol][1]] for m in GVAR}
        g["IPW"] = [round(float(v), 4) for v in np.asarray(pi_ipw, float)[:, GRIDcol][1]]
        g["AIPW"] = [round(float(v), 4) for v in np.asarray(pi_ai, float)[:, GRIDcol][1]]
        try: g["Kallus"] = [round(float(v), 4) for v in kal.predict_kallus(th.theta, dgp.GRID, ("affine",))[:, 1]]
        except Exception: g["Kallus"] = None
        g["Full-info"] = [round(float(v), 4) for v in fitr[:, GRIDcol][1]]; g["Best-means"] = [round(float(v), 4) for v in bmtr[:, GRIDcol][1]]
        grids[gi] = g
        print("seed%d %s Γ=%.2f | R-OW=%.3f IPW=%.3f AIPW=%.3f R-O=%.3f" % (SEED, rname, G, R["methods"]["R-OW"][gi]["rt_test"], m_ipw[1], m_ai[1], R["methods"]["R-O"][gi]["rt_test"]), flush=True)
    R["grids"] = grids
    out["regimes"][rname] = R
out["grid"] = dgp.GRID.tolist(); out["axis"] = dgp.AX.tolist()
OUT = HERE / "_multiseed"; OUT.mkdir(exist_ok=True); (OUT / ("seed%d.json" % SEED)).write_text(json.dumps(out))
print("wrote seed%d.json (2-D, γ=%.1f, matchedΓ=%.2f)" % (SEED, dgp.GAMMA_CONF, MG), flush=True)
