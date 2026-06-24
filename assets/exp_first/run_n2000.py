"""Run the 'first experiment' (Case 4, discrete-X, 2-arm) for ONE seed, in BOTH regimes (UNCAPPED cap=(1,1) and
CAPPED cap=(1,0.5)). 9 methods (R-OW, R-O, IPW, AIPW, R-OW-DR, R-O-DR, Regret-O, Hajek-OW, Kallus) × Γ sweep.
Records realised train/test E[Y], worst-case objective, treat-fraction; + capacity-constrained ceilings (Full-info,
Best-means). On seed 0 saves per-Γ treat-probability grids (π(treat|X)) per method/regime for the interactive chart.
Saves _multiseed/seed{seed}.json.  Usage: python3 run_multiseed.py {seed}"""
import sys, json, importlib.util
import numpy as np
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB
SEED = 0   # single seed for the N=2000 run
NEW = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(NEW)); import common
HERE = NEW / "assets" / "exp_first"
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
K = dgp.K; GAMMAS = [1, 2, 4, 8]; mi = 1; MG = GAMMAS[mi]   # N=2000 run: Γ grid {1,2,4,8}, matched-marker=2
GVAR = ["R-OW", "R-O", "R-OW-DR", "R-O-DR", "Regret-O", "Hajek-OW"]; METHODS = GVAR + ["IPW", "AIPW", "Kallus"]
REGIMES = [("uncap", dgp.CAP_UNCAP), ("cap", dgp.CAP_CAP)]

def constrained_oracle(X, V, cap):
    """max Σ_i Σ_k π_k(x_i)V[i,k] s.t. per-unit simplex, same-X tying, (1/n)Σ_i π_k ≤ cap_k. Returns (K,n)."""
    Xr = np.round(X.ravel(), 6); uniq = np.unique(Xr); n = len(Xr); idx = {u: np.where(Xr == u)[0] for u in uniq}
    Vs = np.array([[V[idx[u], k].sum() for k in range(K)] for u in uniq]); cnt = np.array([len(idx[u]) for u in uniq]); ns = len(uniq)
    m = gp.Model(); m.Params.OutputFlag = 0; pi = m.addVars(ns, K, lb=0, ub=1)
    m.setObjective(gp.quicksum(pi[s, k] * Vs[s, k] for s in range(ns) for k in range(K)), GRB.MAXIMIZE)
    for s in range(ns): m.addConstr(gp.quicksum(pi[s, k] for k in range(K)) == 1)
    for k in range(K): m.addConstr(gp.quicksum(cnt[s] * pi[s, k] for s in range(ns)) <= cap[k] * n)
    m.optimize(); out = np.zeros((K, n))
    for si, u in enumerate(uniq):
        for k in range(K): out[k, idx[u]] = pi[si, k].X
    return out

NTR = 2000
import time as _time
class _Copy:
    def __init__(self, pi): self.pi = pi; self.objective_value = float("nan")
rng = np.random.default_rng(SEED); tr = dgp.generate(NTR, rng); te = dgp.generate(dgp.n_te, rng)
w, _ = common.ipw_weights_from_data(tr.X, tr.T, K); Dm = common.pairwise_distance_matrix(tr.X)
eps = tuple(common.tight_epsilon(Dm, tr.T, w, K, is_distance=True, c_eps=1.0))
muhat = common.outcome_means(tr.X, tr.T, tr.Y, n_arms=K, cross_fit=True)
# deploy maps (exact discrete-X grid lookup)
by = {}
for j, xv in enumerate(np.round(tr.X.ravel(), 6)): by.setdefault(round(float(xv), 6), j)
GRIDcol = np.array([by[round(float(x), 6)] for x in dgp.GRID])
nn_te = np.array([by[round(float(x), 6)] for x in np.round(te.X.ravel(), 6)])
def metrics(pi):
    pi = np.asarray(pi, float)
    rt_tr = float((pi * tr.Ypot.T).sum() / pi.shape[1]); pte = pi[:, nn_te]
    rt_te = float((pte * te.Ypot.T).sum() / pte.shape[1]); treat = float(pi[1].mean())
    return rt_tr, rt_te, treat
def kal_metrics(pi_tr, pi_te):
    rt_tr = float((pi_tr * tr.Ypot).sum(1).mean()); rt_te = float((pi_te * te.Ypot).sum(1).mean())
    return rt_tr, rt_te, float(pi_tr[:, 1].mean())

out = {"seed": SEED, "gammas": GAMMAS, "gamma_conf": dgp.GAMMA_CONF, "matched_gamma": MG, "regimes": {}}
for rname, cap in REGIMES:
    R = {"cap": list(cap), "methods": {m: {} for m in METHODS}}
    # constrained ceilings for this regime
    fi = constrained_oracle(te.X, te.Ypot, cap); bm = constrained_oracle(te.X, te.mu, cap)
    fitr = constrained_oracle(tr.X, tr.Ypot, cap); bmtr = constrained_oracle(tr.X, tr.mu, cap)
    R["ceilings"] = {"full_info_test": float((fi * te.Ypot.T).sum() / te.Ypot.shape[0]),
                     "best_means_test": float((bm * te.mu.T).sum() / te.mu.shape[0]),
                     "full_info_train": float((fitr * tr.Ypot.T).sum() / tr.Ypot.shape[0]),
                     "best_means_train": float((bmtr * tr.mu.T).sum() / tr.mu.shape[0])}
    # Γ-independent: IPW, AIPW (once)
    m_ipw = metrics(ipw(tr.X, tr.T, tr.Y, w, n_arms=K, cap=cap, discretize=False).pi)
    pi_ipw = ipw(tr.X, tr.T, tr.Y, w, n_arms=K, cap=cap, discretize=False).pi
    pi_ai = rodr(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=1.0, cap=cap, discretize=False).pi
    m_ai = metrics(pi_ai)
    grids = {}  # seed0 per-Γ treat-grid
    for gi, G in enumerate(GAMMAS):
        R["methods"]["IPW"][gi] = {"rt_train": m_ipw[0], "rt_test": m_ipw[1], "treat": m_ipw[2], "obj": float("nan")}
        R["methods"]["AIPW"][gi] = {"rt_train": m_ai[0], "rt_test": m_ai[1], "treat": m_ai[2], "obj": float("nan")}
        sv = {}
        sv["R-O"] = ro(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, cap=cap, discretize=False)
        sv["R-O-DR"] = rodr(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=G, cap=cap, discretize=False)
        sv["Regret-O"] = rego(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, maximize=True, cap=cap, discretize=False)
        if G == 1.0:   # W methods at Γ=1 equal their parent X-X method (not re-solved)
            sv["R-OW"] = _Copy(pi_ipw); sv["R-OW-DR"] = _Copy(pi_ai); sv["Hajek-OW"] = _Copy(pi_ipw)
        else:
            _t0 = _time.time()
            sv["R-OW"] = row(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, cap=cap, discretize=False, zscore=False, epsilon=eps)
            sv["R-OW-DR"] = rowdr(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=G, cap=cap, discretize=False, zscore=False, epsilon=eps)
            sv["Hajek-OW"] = regow(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, maximize=True, cap=cap, discretize=False, zscore=False, epsilon=eps)
            print("    %s Γ=%g W-solves done (%.1f min)" % (rname, G, (_time.time()-_t0)/60), flush=True)
        for m in GVAR:
            mm = metrics(sv[m].pi)
            R["methods"][m][gi] = {"rt_train": mm[0], "rt_test": mm[1], "treat": mm[2], "obj": float(getattr(sv[m], "objective_value", float("nan")))}
        # Kallus (free logistic policy; cannot enforce cap -> uncapped regime only is fair, but fit anyway)
        try:
            th = kal.fit_kallus(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(G), maximize=True, wasserstein=False, basis=("affine",), n_iters=20, n_restarts=4, seed=0)
            pit = kal.predict_kallus(th.theta, tr.X, ("affine",)); pie = kal.predict_kallus(th.theta, te.X, ("affine",))
            km = kal_metrics(pit, pie)
            R["methods"]["Kallus"][gi] = {"rt_train": km[0], "rt_test": km[1], "treat": km[2], "obj": float(th.objective_value)}
        except Exception as e:
            R["methods"]["Kallus"][gi] = {"rt_train": None, "rt_test": None, "treat": None, "obj": None}
        if True:  # treat-prob grid per method — saved for ALL seeds so the chart can average policies across seeds
            g = {m: [round(float(v), 4) for v in np.asarray(sv[m].pi, float)[:, GRIDcol][1]] for m in GVAR}
            g["IPW"] = [round(float(v), 4) for v in np.asarray(pi_ipw, float)[:, GRIDcol][1]]
            g["AIPW"] = [round(float(v), 4) for v in np.asarray(pi_ai, float)[:, GRIDcol][1]]
            try: g["Kallus"] = [round(float(v), 4) for v in kal.predict_kallus(th.theta, dgp.GRID.reshape(-1, 1), ("affine",))[:, 1]]
            except Exception: g["Kallus"] = None
            g["Full-info"] = [round(float(v), 4) for v in fitr[:, GRIDcol][1]]; g["Best-means"] = [round(float(v), 4) for v in bmtr[:, GRIDcol][1]]
            grids[gi] = g
        print("seed%d %s Γ=%.2f | R-OW=%.3f IPW=%.3f AIPW=%.3f" % (SEED, rname, G, R["methods"]["R-OW"][gi]["rt_test"], m_ipw[1], m_ai[1]), flush=True)
    R["grids"] = grids
    out["regimes"][rname] = R
out["grid"] = dgp.GRID.tolist()
OUT = HERE / "_n2000"; OUT.mkdir(exist_ok=True); (OUT / ("seed%d.json" % SEED)).write_text(json.dumps(out))
print("wrote seed%d.json (γ=%.2f, matchedΓ=%.2f)" % (SEED, dgp.GAMMA_CONF, MG), flush=True)
