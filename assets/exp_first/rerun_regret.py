"""Regret-only re-run for the 'first experiment' (Case 4): recompute ONLY Regret-O, Hajek-OW, Kallus
with the corrected SELF-NORMALISED Hájek regret, for every existing seed file, both regimes, all Γ,
incl. per-Γ treat grids — and MERGE into _multiseed/seed*.json WITHOUT touching the other 6 methods.
Regret-O & Kallus use RAW inverse weights (floating-denominator self-norm box); Hajek-OW keeps Hájek
weights (its transport mass-balance already pins Σ_{I_k}W=n, i.e. it is already self-normalised).
Usage: python3 rerun_regret.py"""
import sys, json, importlib.util
import numpy as np
from pathlib import Path
NEW = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(NEW)); import common
HERE = NEW / "assets" / "exp_first"
sp = importlib.util.spec_from_file_location("dgp_first", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sys.modules["dgp_first"] = dgp; sp.loader.exec_module(dgp)
def load(n, rel):
    s = importlib.util.spec_from_file_location(n, str(NEW / rel)); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
LPf = lambda rel, fn: getattr(load(fn, rel), fn)
rego = LPf("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")
regow = LPf("methods/Hajek-O-W/Capped/hajek_o_w_capped.py", "solve_hajek_o_w_capped")
kal = load("kal", "methods/Kallus/kallus.py")
ZS = False                                                  # exp_first OW ground cost: raw-X (matches run_multiseed)
K = dgp.K; GAMMAS, mi = dgp.gammas_for(dgp.GAMMA_CONF); REGIMES = [("uncap", dgp.CAP_UNCAP), ("cap", dgp.CAP_CAP)]

seeds = sorted((HERE / "_multiseed").glob("seed*.json"))
print(f"[exp_first] re-running regret methods for {len(seeds)} seeds (ZS={ZS})")
for spath in seeds:
    SEED = int(spath.stem.replace("seed", ""))
    rng = np.random.default_rng(SEED); tr = dgp.generate(dgp.n_tr, rng); te = dgp.generate(dgp.n_te, rng)
    w, P = common.ipw_weights_from_data(tr.X, tr.T, K)            # Hájek (Hajek-OW + Wasserstein radius)
    w_raw = 1.0 / P[np.arange(dgp.n_tr), tr.T]                    # RAW inverse weights (Regret-O, Kallus)
    Dm = common.pairwise_distance_matrix(tr.X)
    eps = tuple(common.tight_epsilon(Dm, tr.T, w, K, is_distance=True, c_eps=1.0))
    by = {}
    for j, xv in enumerate(np.round(tr.X.ravel(), 6)): by.setdefault(round(float(xv), 6), j)
    GRIDcol = np.array([by[round(float(x), 6)] for x in dgp.GRID])
    nn_te = np.array([by[round(float(x), 6)] for x in np.round(te.X.ravel(), 6)])
    def metrics(pi):
        pi = np.asarray(pi, float); rt_tr = float((pi * tr.Ypot.T).sum() / pi.shape[1])
        pte = pi[:, nn_te]; rt_te = float((pte * te.Ypot.T).sum() / pte.shape[1]); return rt_tr, rt_te, float(pi[1].mean())
    def grid_of(pi): return [round(float(v), 4) for v in np.asarray(pi, float)[:, GRIDcol][1]]
    data = json.loads(spath.read_text())
    for rname, cap in REGIMES:
        R = data["regimes"][rname]
        for gi, G in enumerate(GAMMAS):
            svO = rego(tr.X, tr.T, tr.Y, w_raw, n_arms=K, Gamma=G, maximize=True, cap=cap, discretize=False)
            mO = metrics(svO.pi); R["methods"]["Regret-O"][str(gi)] = {"rt_train": mO[0], "rt_test": mO[1], "treat": mO[2], "obj": float(svO.objective_value)}; R["grids"][str(gi)]["Regret-O"] = grid_of(svO.pi)
            svW = regow(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, maximize=True, cap=cap, discretize=False, zscore=ZS, epsilon=eps)
            mW = metrics(svW.pi); R["methods"]["Hajek-OW"][str(gi)] = {"rt_train": mW[0], "rt_test": mW[1], "treat": mW[2], "obj": float(svW.objective_value)}; R["grids"][str(gi)]["Hajek-OW"] = grid_of(svW.pi)
            try:
                th = kal.fit_kallus(tr.X, tr.T, tr.Y, w_raw, n_arms=K, Gamma=float(G), maximize=True, wasserstein=False, basis=("affine",), n_iters=20, n_restarts=4, seed=0)
                pit = kal.predict_kallus(th.theta, tr.X, ("affine",)); pie = kal.predict_kallus(th.theta, te.X, ("affine",))
                R["methods"]["Kallus"][str(gi)] = {"rt_train": float((pit * tr.Ypot).sum(1).mean()), "rt_test": float((pie * te.Ypot).sum(1).mean()), "treat": float(pit[:, 1].mean()), "obj": float(th.objective_value)}
                R["grids"][str(gi)]["Kallus"] = [round(float(v), 4) for v in kal.predict_kallus(th.theta, dgp.GRID.reshape(-1, 1), ("affine",))[:, 1]]
            except Exception as e:
                R["methods"]["Kallus"][str(gi)] = {"rt_train": None, "rt_test": None, "treat": None, "obj": None}; R["grids"][str(gi)]["Kallus"] = None
        print(f"  seed{SEED} {rname}: Regret-O@mΓ={R['methods']['Regret-O'][str(mi)]['rt_test']:.3f} Hajek-OW={R['methods']['Hajek-OW'][str(mi)]['rt_test']:.3f} Kallus={R['methods']['Kallus'][str(mi)]['rt_test']:.3f}", flush=True)
    spath.write_text(json.dumps(data)); print(f"  merged seed{SEED}", flush=True)
print("done.")
