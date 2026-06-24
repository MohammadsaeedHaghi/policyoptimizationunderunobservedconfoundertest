"""Run all XX and OX methods on the 'first prime' DGP (piecewise-linear first experiment), N_train=1500, one seed,
BOTH regimes (uncapped cap=(1,1) and capped cap=(1,0.5)).  Methods:
  XX (Γ-free):  IPW-X-X, DoublyRobust-X-X, Direct-X-X
  OX (Γ-sweep): IPW-O-X, DoublyRobust-O-X, Hajek-O-X
Records realised test E[Y] + treat fraction per Γ, plus deployed policy π(treat|X) for the policy plot, plus the
constrained best-means oracle ceiling.  Saves prime_results.json and splices the value+policy charts into index.html
(color = estimator family, shape = uncertainty set; XX dashed/no-marker, OX squares).  No Wasserstein here."""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import gurobipy as gp
from gurobipy import GRB
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("pdgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sys.modules["pdgp"] = dgp; sp.loader.exec_module(dgp)
def Lf(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
SOLV = {
  "uncap": {"IPW-X-X": Lf("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped"),
            "DoublyRobust-X-X": Lf("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped"),
            "Direct-X-X": Lf("methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py", "solve_direct_x_x_uncapped"),
            "IPW-O-X": Lf("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped"),
            "DoublyRobust-O-X": Lf("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped"),
            "Hajek-O-X": Lf("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py", "solve_hajek_o_x_uncapped")},
  "cap":   {"IPW-X-X": Lf("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped"),
            "DoublyRobust-X-X": Lf("methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py", "solve_doublyrobust_x_x_capped"),
            "Direct-X-X": Lf("methods/Direct-X-X/Capped/direct_x_x_capped.py", "solve_direct_x_x_capped"),
            "IPW-O-X": Lf("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped"),
            "DoublyRobust-O-X": Lf("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped"),
            "Hajek-O-X": Lf("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")},
}
K = dgp.K
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]; OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
NTR = 1500; NTE = 20000
rng = np.random.default_rng(0); tr = dgp.generate(NTR, rng); te = dgp.generate(NTE, rng)

# matched Γ = max odds-ratio induced by the unobserved S (e(X,1) vs e(X,0))
e0 = dgp.propensity(dgp.GRID, 0); e1 = dgp.propensity(dgp.GRID, 1)
orr = (e1 / (1 - e1)) / (e0 / (1 - e0)); MG = round(float(np.max(orr)), 2)
GAMMAS = sorted(set([1.0, 2.0, 3.0, 4.0, MG, 6.0, 8.0]));
print("matched Γ (max S odds-ratio) = %.2f ; GAMMAS=%s" % (MG, GAMMAS), flush=True)

w, _ = common.ipw_weights_from_data(tr.X, tr.T, K)
muhat = common.outcome_means(tr.X, tr.T, tr.Y, n_arms=K, cross_fit=True)
by = {}
for j, xv in enumerate(np.round(tr.X.ravel(), 6)): by.setdefault(round(float(xv), 6), j)
GRIDcol = np.array([by[round(float(x), 6)] for x in dgp.GRID])          # training-col index per grid level
nn_te = np.array([by[round(float(x), 6)] for x in np.round(te.X.ravel(), 6)])
def metrics(pi):
    pi = np.asarray(pi, float); pte = pi[:, nn_te]
    return float((pte * te.Ypot.T).sum() / pte.shape[1]), float(pi[1].mean())
def polgrid(pi): return [round(float(pi[1, GRIDcol[i]]), 4) for i in range(len(dgp.GRID))]

def constrained_oracle(X, V, cap):
    Xr = np.round(np.asarray(X).ravel(), 6); levels = np.array(sorted(set(Xr)))
    idx = {round(float(c), 6): s for s, c in enumerate(levels)}; ns = len(levels); n = len(Xr)
    cnt = np.zeros(ns);
    for x in Xr: cnt[idx[round(float(x), 6)]] += 1
    Vs = np.zeros((ns, K))
    for j, x in enumerate(Xr): Vs[idx[round(float(x), 6)]] += V[j]
    m = gp.Model("orc"); m.Params.OutputFlag = 0
    pi = m.addVars(ns, K, lb=0, ub=1)
    for s in range(ns): m.addConstr(gp.quicksum(pi[s, k] for k in range(K)) == 1)
    for k in range(K): m.addConstr(gp.quicksum(cnt[s] * pi[s, k] for s in range(ns)) <= cap[k] * n)
    m.setObjective(gp.quicksum(Vs[s, k] * pi[s, k] for s in range(ns) for k in range(K)), GRB.MAXIMIZE); m.optimize()
    P = np.array([[pi[s, k].X for k in range(K)] for s in range(ns)])
    full = np.array([P[idx[round(float(x), 6)]] for x in Xr])
    return full.T  # (K, n)

out = {"N_train": NTR, "N_test": NTE, "gammas": GAMMAS, "matched_gamma": MG, "grid": [float(v) for v in dgp.GRID], "regimes": {}}
for rname, cap in (("uncap", dgp.CAP_UNCAP), ("cap", dgp.CAP_CAP)):
    bm = constrained_oracle(te.X, te.mu, cap); orc_val = float((bm * te.mu.T).sum() / bm.shape[1])
    R = {"cap": list(cap), "oracle": round(orc_val, 4), "value": {}, "treat": {}, "policy": {}}
    S = SOLV[rname]; kw = {} if rname == "uncap" else {"cap": cap}
    # XX (Γ-free)
    r_ipw = S["IPW-X-X"](tr.X, tr.T, tr.Y, w, n_arms=K, discretize=False, **kw)
    r_dr = S["DoublyRobust-X-X"](tr.X, tr.T, tr.Y, w, muhat, n_arms=K, discretize=False, **kw)
    r_dir = S["Direct-X-X"](tr.X, tr.T, tr.Y, n_arms=K, discretize=False, **kw)
    for nm, res in (("IPW-X-X", r_ipw), ("DoublyRobust-X-X", r_dr), ("Direct-X-X", r_dir)):
        v, tf = metrics(res.pi); R["value"][nm] = [round(v, 4)] * len(GAMMAS); R["treat"][nm] = round(tf, 3)
        R["policy"][nm] = {"_flat": polgrid(res.pi)}
    # OX (Γ sweep)
    for nm in OX: R["value"][nm] = []; R["treat"][nm] = {}; R["policy"][nm] = {}
    for g in GAMMAS:
        gk = str(int(g)) if g == int(g) else str(g)
        ro = S["IPW-O-X"](tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(g), discretize=False, **kw)
        rod = S["DoublyRobust-O-X"](tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=float(g), discretize=False, **kw)
        rh = S["Hajek-O-X"](tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(g), maximize=True, discretize=False, **kw)
        for nm, res in (("IPW-O-X", ro), ("DoublyRobust-O-X", rod), ("Hajek-O-X", rh)):
            v, tf = metrics(res.pi); R["value"][nm].append(round(v, 4)); R["treat"][nm][gk] = round(tf, 3)
            R["policy"][nm][gk] = polgrid(res.pi)
    out["regimes"][rname] = R
    print("[%s] oracle=%.4f | IPW-X-X=%.4f DR-X-X=%.4f Direct-X-X=%.4f | IPW-O-X=%s" %
          (rname, orc_val, R["value"]["IPW-X-X"][0], R["value"]["DoublyRobust-X-X"][0], R["value"]["Direct-X-X"][0], R["value"]["IPW-O-X"]), flush=True)
(HERE / "prime_results.json").write_text(json.dumps(out, indent=2))
print("saved prime_results.json", flush=True)
