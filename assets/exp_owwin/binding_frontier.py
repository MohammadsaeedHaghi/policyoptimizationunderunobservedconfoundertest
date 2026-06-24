"""Binding frontier Γ*(ε): the odds-box Γ at which the box stops binding given a Wasserstein radius ε.
For each ε and each covariate level, the W-ball permits a max treated region-mass M_W; the box with parameter Γ
permits region-mass (n_region + Γ·Σ(ŵ-1))/n. The crossover Γ*_level solves these equal. The box becomes
non-binding (W takes over) once Γ >= Γ*(ε) = max over levels. Below the curve: box binds. Above: W binds.
owwin DGP, seed 0, N=500, treated arm. Saves owwin_binding.json."""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np, gurobipy as gp
from gurobipy import GRB
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("owdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["owdgp"] = d; sp.loader.exec_module(d)
K = d.K; LV = d.LEVELS
obs, _ = d.generate(500, 0); X = obs["X"].ravel(); T = obs["T"]
w, _ = common.ipw_weights_from_data(obs["X"], T, K)            # nominal ŵ = 1/ê
Dm = common.pairwise_distance_matrix(obs["X"])
k = 1; Ik = np.where(T == k)[0]; n = len(T); ns = len(Ik)
D = Dm[np.ix_(np.arange(n), Ik)]
lvl = {round(float(c), 6): li for li, c in enumerate(LV)}
supp_lvl = np.array([lvl[round(float(X[j]), 6)] for j in Ik])
what = w[Ik]                                                    # nominal weights of treated units
levels_present = sorted(set(supp_lvl))
tight = common.tight_epsilon(Dm, T, w, K, is_distance=True, c_eps=1.0)[k]
EPS = [round(tight * f, 5) for f in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]]

def maxmass(L0, eps):
    m = gp.Model(); m.Params.OutputFlag = 0
    z = m.addVars(n, ns, lb=0.0); W = m.addVars(ns, lb=0.0)
    for i in range(n): m.addConstr(gp.quicksum(z[i, j] for j in range(ns)) == 1.0 / n)
    for j in range(ns): m.addConstr(gp.quicksum(z[i, j] for i in range(n)) == W[j] / n)
    m.addConstr(gp.quicksum(D[i, j] * z[i, j] for i in range(n) for j in range(ns)) <= eps)
    m.setObjective(gp.quicksum(W[j] for j in range(ns) if supp_lvl[j] == L0), GRB.MAXIMIZE)  # total mass Σ_region W_j
    m.optimize(); return m.ObjVal                                # = Σ_{region} W_j
rows = []; t0 = time.time()
for eps in EPS:
    gstar = 0.0; arg = None
    for L0 in levels_present:
        S = float(np.sum(what[supp_lvl == L0] - 1.0)); nreg = int(np.sum(supp_lvl == L0))
        if S <= 1e-9: continue
        MW = maxmass(L0, eps)                                    # Σ_region W_j permitted by W-ball
        g = (MW - nreg) / S                                      # box Γ giving same region mass
        if g > gstar: gstar = g; arg = float(LV[L0])
    rows.append({"eps": eps, "gamma_star": round(gstar, 3), "level": arg})
    print("ε=%.4f -> Γ*=%.2f (level X=%s)  (%.1f min)" % (eps, gstar, arg, (time.time() - t0) / 60), flush=True)
(HERE / "owwin_binding.json").write_text(json.dumps({"tight_eps": round(float(tight), 5), "rows": rows}, indent=2))
print("saved owwin_binding.json", flush=True)
