"""Regret-only re-run for the 'second experiment' (2-D): recompute ONLY Regret-O, Hajek-OW, Kallus
with the corrected SELF-NORMALISED Hájek regret, for every existing seed file, both regimes, all Γ,
incl. the per-(method,Γ) treat grids over the 11×11=121 grid — MERGED into _multiseed/seed*.json
WITHOUT touching the other 6 methods. Regret-O & Kallus use RAW inverse weights (floating-denominator
self-norm box); Hajek-OW keeps Hájek weights (its transport mass-balance already self-normalises).
Wasserstein ground cost uses zscore=True (2-D), matching run_multiseed.py. Usage: python3 rerun_regret.py"""
import sys, json, importlib.util
import numpy as np
from pathlib import Path
NEW = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(NEW)); import common
HERE = NEW / "assets" / "exp_second"
sp = importlib.util.spec_from_file_location("dgp_second", str(HERE / "dgp.py")); dgp = importlib.util.module_from_spec(sp); sys.modules["dgp_second"] = dgp; sp.loader.exec_module(dgp)
def load(n, rel):
    s = importlib.util.spec_from_file_location(n, str(NEW / rel)); m = importlib.util.module_from_spec(s); sys.modules[n] = m; s.loader.exec_module(m); return m
LPf = lambda rel, fn: getattr(load(fn, rel), fn)
rego = LPf("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")
regow = LPf("methods/Hajek-O-W/Capped/hajek_o_w_capped.py", "solve_hajek_o_w_capped")
kal = load("kal", "methods/Kallus/kallus.py")
K = dgp.K; GAMMAS = dgp.GAMMAS; mi = dgp.mi; ZS = True       # 2-D ground cost -> zscore ON
REGIMES = [("uncap", dgp.CAP_UNCAP), ("cap", dgp.CAP_CAP)]

seeds = sorted((HERE / "_multiseed").glob("seed*.json"))
print(f"[exp_second] re-running regret methods for {len(seeds)} seeds (ZS={ZS})", flush=True)
for spath in seeds:
    SEED = int(spath.stem.replace("seed", ""))
    rng = np.random.default_rng(SEED); tr = dgp.generate(dgp.n_tr, rng); te = dgp.generate(dgp.n_te, rng)
    w, P = common.ipw_weights_from_data(tr.X, tr.T, K)            # Hájek (Hajek-OW)
    w_raw = 1.0 / P[np.arange(dgp.n_tr), tr.T]                    # RAW inverse weights (Regret-O, Kallus)
    # Hajek-OW uses zscore=True; let the solver compute eps INTERNALLY with the z-scored D (matches run_multiseed.py).
    # Passing a non-z-scored eps here is a geometry mismatch and makes Γ=1 infeasible.
    GRIDcol = np.argmin(((dgp.GRID[:, None, :] - tr.X[None, :, :]) ** 2).sum(2), axis=1)   # (121,)
    nn_te = np.argmin(((te.X[:, None, :] - tr.X[None, :, :]) ** 2).sum(2), axis=1)          # (n_te,)
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
            svW = regow(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, maximize=True, cap=cap, discretize=False, zscore=ZS)
            mW = metrics(svW.pi); R["methods"]["Hajek-OW"][str(gi)] = {"rt_train": mW[0], "rt_test": mW[1], "treat": mW[2], "obj": float(svW.objective_value)}; R["grids"][str(gi)]["Hajek-OW"] = grid_of(svW.pi)
            try:
                th = kal.fit_kallus(tr.X, tr.T, tr.Y, w_raw, n_arms=K, Gamma=float(G), maximize=True, wasserstein=False, basis=("affine",), n_iters=20, n_restarts=4, seed=0)
                pit = kal.predict_kallus(th.theta, tr.X, ("affine",)); pie = kal.predict_kallus(th.theta, te.X, ("affine",))
                R["methods"]["Kallus"][str(gi)] = {"rt_train": float((pit * tr.Ypot).sum(1).mean()), "rt_test": float((pie * te.Ypot).sum(1).mean()), "treat": float(pit[:, 1].mean()), "obj": float(th.objective_value)}
                R["grids"][str(gi)]["Kallus"] = [round(float(v), 4) for v in kal.predict_kallus(th.theta, dgp.GRID, ("affine",))[:, 1]]
            except Exception as e:
                R["methods"]["Kallus"][str(gi)] = {"rt_train": None, "rt_test": None, "treat": None, "obj": None}; R["grids"][str(gi)]["Kallus"] = None
        print(f"  seed{SEED} {rname}: Regret-O@mΓ={R['methods']['Regret-O'][str(mi)]['rt_test']:.3f} Hajek-OW={R['methods']['Hajek-OW'][str(mi)]['rt_test']:.3f} Kallus={R['methods']['Kallus'][str(mi)]['rt_test']:.3f}", flush=True)
    spath.write_text(json.dumps(data)); print(f"  merged seed{SEED}", flush=True)
print("done.")
