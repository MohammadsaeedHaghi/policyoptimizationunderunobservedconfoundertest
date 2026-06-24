"""What odds-ratio bound does the Wasserstein radius ε imply? On discrete X, transport WITHIN a covariate level is
free (distance 0), so the W-ball does NOT bound per-unit weights — it bounds the per-LEVEL (covariate-region) mass.
For each ε we compute, via LP, the max factor by which the W-ball can inflate the treated weight-mass on a single
X-level, relative to its calibrated-uniform share — i.e. the effective odds-ratio Γ_eff(ε) at the covariate-region
scale. owwin DGP, seed 0, N=700, treated arm."""
import sys, importlib.util
from pathlib import Path
import numpy as np, gurobipy as gp
from gurobipy import GRB
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("owdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["owdgp"] = d; sp.loader.exec_module(d)
K = d.K; LV = d.LEVELS
obs, _ = d.generate(700, 0); X = obs["X"].ravel(); T = obs["T"]
w, _ = common.ipw_weights_from_data(obs["X"], T, K)
Dm = common.pairwise_distance_matrix(obs["X"])
k = 1; Ik = np.where(T == k)[0]; n = len(T)                       # treated arm
D = Dm[np.ix_(np.arange(n), Ik)]                                  # demand(all n) x supply(treated)
# nominal calibrated share of treated mass per level (uniform demand 1/n -> level ℓ gets n_ℓ/n of demand)
lvl_of = {round(float(c), 6): li for li, c in enumerate(LV)}
supp_lvl = np.array([lvl_of[round(float(X[j]), 6)] for j in Ik])
print("treated arm: n=%d  |I_k|=%d  levels with treated: %s" % (n, len(Ik), sorted(set(supp_lvl))))
for ce in [0.5, 1.0, 2.0, 4.0]:
    eps = common.tight_epsilon(Dm, T, w, K, is_distance=True, c_eps=ce)[k]
    best = 0.0; bestlvl = None
    for L0 in sorted(set(supp_lvl)):
        m = gp.Model(); m.Params.OutputFlag = 0
        z = m.addVars(n, len(Ik), lb=0.0)
        W = m.addVars(len(Ik), lb=0.0)
        for i in range(n): m.addConstr(gp.quicksum(z[i, j] for j in range(len(Ik))) == 1.0 / n)
        for j in range(len(Ik)): m.addConstr(gp.quicksum(z[i, j] for i in range(n)) == W[j] / n)
        m.addConstr(gp.quicksum(D[i, j] * z[i, j] for i in range(n) for j in range(len(Ik))) <= eps)
        # maximize treated mass on level L0  (fraction of total = (1/n)Σ_{j in L0} W_j)
        m.setObjective(gp.quicksum(W[j] for j in range(len(Ik)) if supp_lvl[j] == L0) / n, GRB.MAXIMIZE)
        m.optimize()
        frac = m.ObjVal
        if frac > best: best = frac; bestlvl = L0
    # nominal share of THIS level under uniform demand = n_ℓ_all / n (all units at that level)
    nlvlall = np.mean([lvl_of[round(float(x), 6)] == bestlvl for x in X])
    Geff = best / nlvlall if nlvlall > 0 else float("inf")
    print("c_eps=%.1f  ε=%.4f | max treated-mass on a level=%.3f (level X=%.2f, baseline share=%.3f) -> inflation Γ_eff≈%.1fx"
          % (ce, eps, best, LV[bestlvl], nlvlall, Geff))
print("\nNote: WITHIN a level distance=0, so the W-ball places NO bound on per-unit weights (intra-level transport is free);")
print("it only bounds the per-LEVEL (covariate-region) mass. The numbers above are that level-scale implied bound.")
