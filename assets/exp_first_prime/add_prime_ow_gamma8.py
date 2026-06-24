"""Solve IPW-O-W and DoublyRobust-O-W at Γ=8 ONLY, for the first-prime N_train=1500 study, and merge into the saved
prime_ow_<regime>_results.json (keeping the existing Γ∈{1,4,7.46} points). Does NOT touch index.html (patched
separately to avoid a write race). Usage: python3 add_prime_ow_gamma8.py {uncap|cap}"""
import sys, json, importlib.util, time
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("pdgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sys.modules["pdgp"] = dgp; sp.loader.exec_module(dgp)
def Lf(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
REG = sys.argv[1]
assert REG in ("uncap", "cap")
if REG == "cap":
    ipwow = Lf("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped")
    drow = Lf("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped")
    CAP = dgp.CAP_CAP; kw = {"cap": CAP}
else:
    ipwow = Lf("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
    drow = Lf("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")
    kw = {}
K = dgp.K; NTR, NTE = 1500, 20000; GAM = 8.0
rng = np.random.default_rng(0); tr = dgp.generate(NTR, rng); te = dgp.generate(NTE, rng)
w, _ = common.ipw_weights_from_data(tr.X, tr.T, K)
muhat = common.outcome_means(tr.X, tr.T, tr.Y, n_arms=K, cross_fit=True)
by = {}
for j, xv in enumerate(np.round(tr.X.ravel(), 6)): by.setdefault(round(float(xv), 6), j)
GRIDcol = np.array([by[round(float(x), 6)] for x in dgp.GRID])
nn_te = np.array([by[round(float(x), 6)] for x in np.round(te.X.ravel(), 6)])
def metrics(pi):
    pi = np.asarray(pi, float); pte = pi[:, nn_te]
    return float((pte * te.Ypot.T).sum() / pte.shape[1])
def polgrid(pi): return [round(float(pi[1, GRIDcol[i]]), 4) for i in range(len(dgp.GRID))]
Dm = common.pairwise_distance_matrix(tr.X)
eps = tuple(common.tight_epsilon(Dm, tr.T, w, K, is_distance=True, c_eps=1.0))
print("[%s] ε=%.5f,%.5f  solving Γ=8 ..." % (REG, eps[0], eps[1]), flush=True)
t0 = time.time()
ro = ipwow(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=GAM, discretize=False, zscore=False, epsilon=eps, **kw)
rd = drow(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=GAM, discretize=False, zscore=False, epsilon=eps, **kw)
vo, vd = round(metrics(ro.pi), 4), round(metrics(rd.pi), 4)
print("[%s] Γ=8  IPW-O-W=%.4f  DR-O-W=%.4f  (%.1f min)" % (REG, vo, vd, (time.time() - t0) / 60), flush=True)

# merge into saved results json (index of Γ=8 in the grid)
res = json.loads((HERE / ("prime_ow_%s_results.json" % REG)).read_text())
G = res["gammas"]; i8 = G.index(8.0)
res["value"]["IPW-O-W"][i8] = vo; res["value"]["DoublyRobust-O-W"][i8] = vd
res["policy"]["IPW-O-W"]["8"] = polgrid(ro.pi); res["policy"]["DoublyRobust-O-W"]["8"] = polgrid(rd.pi)
(HERE / ("prime_ow_%s_results.json" % REG)).write_text(json.dumps(res, indent=2))
print("[%s] merged Γ=8 into prime_ow_%s_results.json" % (REG, REG), flush=True)
