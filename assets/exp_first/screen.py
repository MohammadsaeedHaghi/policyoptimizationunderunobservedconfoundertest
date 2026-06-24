"""Screen γ ∈ {2,2.5,3} for the 'first experiment' DGP: for each γ and each regime (UNCAPPED cap=(1,1) /
CAPPED cap=(1,0.5)) fit R-OW, R-OW-DR, IPW, AIPW at the matched Γ over a few seeds, deploy on a fresh test set,
and report realised E[Y]. Goal: pick γ where (a) R-OW > IPW and (b) X plays a real role (already checked: γ<=3).
Usage: python3 screen.py"""
import sys, importlib.util
import numpy as np
from pathlib import Path
NEW = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(NEW)); import common
HERE = NEW / "assets" / "exp_first"
sp = importlib.util.spec_from_file_location("dgp", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sp.loader.exec_module(dgp)
def load(n, rel):
    s = importlib.util.spec_from_file_location(n, str(NEW / rel)); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
LPf = lambda rel, fn: getattr(load(fn, rel), fn)
row = LPf("methods/IPW-O-W/Capped/ipw_o_w_capped.py", "solve_ipw_o_w_capped")
ipw = LPf("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped")
aipw = LPf("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped")
rowdr = LPf("methods/DoublyRobust-O-W/Capped/doublyrobust_o_w_capped.py", "solve_doublyrobust_o_w_capped")
K = dgp.K; SEEDS = [0, 1, 2]
def deploy_value(pi, tr, te):
    pi = np.asarray(pi, float)
    by = {}
    for j, xv in enumerate(np.round(tr.X.ravel(), 6)): by.setdefault(round(float(xv), 6), j)
    nn = np.array([by[round(float(x), 6)] for x in np.round(te.X.ravel(), 6)])
    c = pi[:, nn]; return float((c * te.Ypot.T).sum() / c.shape[1])
for gamma in (2.0, 2.5, 3.0):
    mg = round(float(np.exp(gamma / 2.0)), 4)
    for regime, cap in (("UNCAP", dgp.CAP_UNCAP), ("CAP.5", dgp.CAP_CAP)):
        acc = {m: [] for m in ["R-OW", "R-OW-DR", "IPW", "AIPW"]}; treatfrac = {m: [] for m in acc}
        for sd in SEEDS:
            rng = np.random.default_rng(sd); tr = dgp.generate(dgp.n_tr, rng, gamma=gamma); te = dgp.generate(dgp.n_te, rng, gamma=gamma)
            w, _ = common.ipw_weights_from_data(tr.X, tr.T, K); Dm = common.pairwise_distance_matrix(tr.X)
            eps = tuple(common.tight_epsilon(Dm, tr.T, w, K, is_distance=True, c_eps=1.0))
            muhat = common.outcome_means(tr.X, tr.T, tr.Y, n_arms=K, cross_fit=True)
            P = {}
            P["R-OW"] = row(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=mg, cap=cap, discretize=False, zscore=False, epsilon=eps).pi
            P["R-OW-DR"] = rowdr(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=mg, cap=cap, discretize=False, zscore=False, epsilon=eps).pi
            P["IPW"] = ipw(tr.X, tr.T, tr.Y, w, n_arms=K, cap=cap, discretize=False).pi
            P["AIPW"] = aipw(tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=1.0, cap=cap, discretize=False).pi
            for m in acc:
                acc[m].append(deploy_value(P[m], tr, te)); treatfrac[m].append(float(np.asarray(P[m], float)[1].mean()))
        def fmt(m): return "%s=%.3f(treat %.0f%%)" % (m, np.mean(acc[m]), 100 * np.mean(treatfrac[m]))
        edge = np.mean(acc["R-OW"]) - np.mean(acc["IPW"])
        print("γ=%.1f mΓ=%.2f %s | %s | %s | %s | %s || R-OW−IPW=%+.3f %s"
              % (gamma, mg, regime, fmt("R-OW"), fmt("R-OW-DR"), fmt("IPW"), fmt("AIPW"), edge, "✓" if edge > 0.003 else ""), flush=True)
print("\noracle: never=0.719 treat-all=0.762 optimal(treatX>=0)=0.817")
