"""Regret-only re-run for exp_1d: recompute ONLY Regret-O, Hajek-OW, Kallus with the corrected
SELF-NORMALISED Hájek regret (box built on RAW inverse weights), for every existing seed file, both
regimes, all Γ, including the per-Γ treat grids — and MERGE them into _multiseed/{mode}_seed*.json
WITHOUT touching the other 6 methods (which are already correct). Replicates the exact DGP/deploy/
metrics of exp_1d_multiseed.py. Usage: python3 rerun_regret.py [bern]"""
import sys, importlib.util, json
import numpy as np
from pathlib import Path
MODE = sys.argv[1] if len(sys.argv) > 1 else "bern"
NEW = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
sys.path.insert(0, str(NEW)); import common
ASSETS = (NEW / "assets" / "exp_1d") if MODE == "bern" else (NEW / "assets" / "exp_1d_uniformS")
OUT = ASSETS / "_multiseed"
def load(n, rel):
    sp = importlib.util.spec_from_file_location(n, str(NEW / rel)); m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m); return m
A, C, B = 4.0, 5.0, 1.0; theta0, theta1 = 0.5, -1.2; K = 2
REGIMES = [("uncap", (1.0, 1.0)), ("cap", (1.0, 0.5))]
GRID = np.round(np.linspace(-1, 1, 21), 6); S_GRID = np.round(np.arange(0, 1.0001, 0.1), 6); n_tr, n_te = 500, 2000; gt = 5.0
GAMMAS = [1.0, 3.0, 6.0, 9.0, round(float(np.exp(gt / 2)), 4), 16.0]
def sig(z): z = np.asarray(z, float); return 1 / (1 + np.exp(-np.clip(z, -40, 40)))
def pa(X, S, k): return np.clip(sig(theta0 + 0 * np.asarray(X, float)) if k == 0 else sig(theta1 + A * np.asarray(X, float) + C * np.asarray(S, float)), 1e-3, 1 - 1e-3)
def generate(nn, gamma, rng):
    X = rng.choice(GRID, size=nn)
    S = (rng.uniform(size=nn) < 0.5).astype(int) if MODE == "bern" else rng.choice(S_GRID, size=nn)
    Yp = np.column_stack([(rng.uniform(size=nn) < pa(X, S, 0)).astype(float), (rng.uniform(size=nn) < pa(X, S, 1)).astype(float)])
    U1 = -B * X + gamma * (S - 0.5); p1 = np.exp(U1) / (1 + np.exp(U1)); T = (rng.uniform(size=nn) < p1).astype(int)
    from types import SimpleNamespace
    return SimpleNamespace(X=X.reshape(-1, 1), T=T, Y=Yp[np.arange(nn), T], Ypot=Yp)
LPf = lambda rel, fn: getattr(load(fn, rel), fn)
rego = LPf("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")
regow = LPf("methods/Hajek-O-W/Capped/hajek_o_w_capped.py", "solve_hajek_o_w_capped")
kal = load("kal", "methods/Kallus/kallus.py")

seeds = sorted(OUT.glob(f"{MODE}_seed*.json"))
print(f"[{MODE}] re-running regret methods for {len(seeds)} seeds")
for sp in seeds:
    SEED = int(sp.stem.split("seed")[1])
    rng = np.random.default_rng(SEED); tr = generate(n_tr, gt, rng); te = generate(n_te, gt, rng)
    w, P = common.ipw_weights_from_data(tr.X, tr.T, K)             # Hájek weights: Hajek-OW (transport calibrates) + eps
    w_raw = 1.0 / P[np.arange(n_tr), tr.T]                         # RAW inverse weights (≥1) for the self-norm box (O, Kallus)
    Dm = common.pairwise_distance_matrix(tr.X)
    eps = tuple(common.tight_epsilon(Dm, tr.T, w, K, is_distance=True, c_eps=1.0))
    uniq = {}
    for j, xv in enumerate(np.round(tr.X.ravel(), 6)): uniq.setdefault(float(xv), j)
    ukeys = np.array(list(uniq.keys())); uidx = np.array([uniq[k] for k in ukeys])
    nn_te = np.array([int(np.argmin(np.abs(ukeys - x))) for x in np.round(te.X.ravel(), 6)])
    nn_tr = np.array([int(np.argmin(np.abs(ukeys - x))) for x in np.round(tr.X.ravel(), 6)])
    grid_nn = np.array([int(np.argmin(np.abs(ukeys - x))) for x in GRID])
    def rt_test(pi): pi = np.asarray(pi, float); c = pi[:, uidx][:, nn_te]; return float((c * te.Ypot.T).sum() / c.shape[1])
    def rt_train(pi): pi = np.asarray(pi, float); c = pi[:, uidx][:, nn_tr]; return float((c * tr.Ypot.T).sum() / c.shape[1])
    def treat_frac(pi): return float(np.asarray(pi, float)[1].mean())
    def treat_grid(pi): pi = np.asarray(pi, float); return [round(float(v), 4) for v in pi[:, uidx][1, grid_nn]]
    def solve_regret(name, G, CAP):
        if name == "RegretO":  return rego(tr.X, tr.T, tr.Y, w_raw, n_arms=K, Gamma=G, cap=CAP, maximize=True, discretize=False)
        # Hajek-OW: Hájek weights — the per-arm transport mass-balance already pins Σ_{I_k}W=n (self-normalised).
        if name == "RegretOW": return regow(tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=G, cap=CAP, maximize=True, discretize=False, zscore=False, epsilon=eps)
    data = json.loads(sp.read_text())
    for rname, CAP in REGIMES:
        R = data["regimes"][rname]
        for gi, G in enumerate(GAMMAS):
            for name in ("RegretO", "RegretOW"):
                try:
                    sv = solve_regret(name, float(G), CAP)
                    R["methods"][name][str(gi)] = {"rt_train": rt_train(sv.pi), "rt_test": rt_test(sv.pi), "treat": treat_frac(sv.pi), "obj": float(sv.objective_value)}
                    R["grids"][str(gi)][name] = treat_grid(sv.pi)
                except Exception as e:
                    R["methods"][name][str(gi)] = {"rt_train": None, "rt_test": None, "treat": None, "obj": None}; R["grids"][str(gi)][name] = None
                    print(f"  seed{SEED} {rname} {name} Γ={G}: FAIL {e}")
            try:
                th = kal.fit_kallus(tr.X, tr.T, tr.Y, w_raw, n_arms=K, Gamma=float(G), maximize=True, wasserstein=False, basis=("affine",), n_iters=20, n_restarts=4, seed=0)
                pi_tr = kal.predict_kallus(th.theta, tr.X, ("affine",)); pi_te = kal.predict_kallus(th.theta, te.X, ("affine",))
                R["methods"]["Kallus"][str(gi)] = {"rt_train": float((pi_tr * tr.Ypot).sum(1).mean()), "rt_test": float((pi_te * te.Ypot).sum(1).mean()), "treat": float(pi_tr[:, 1].mean()), "obj": float(th.objective_value)}
                R["grids"][str(gi)]["Kallus"] = [round(float(v), 4) for v in kal.predict_kallus(th.theta, GRID.reshape(-1, 1), ("affine",))[:, 1]]
            except Exception as e:
                R["methods"]["Kallus"][str(gi)] = {"rt_train": None, "rt_test": None, "treat": None, "obj": None}; R["grids"][str(gi)]["Kallus"] = None
                print(f"  seed{SEED} {rname} Kallus Γ={G}: FAIL {e}")
        print(f"  seed{SEED} {rname}: RegretO@mΓ={R['methods']['RegretO']['4']['rt_test']:.3f} RegretOW={R['methods']['RegretOW']['4']['rt_test']:.3f} Kallus={R['methods']['Kallus']['4']['rt_test']:.3f}")
    sp.write_text(json.dumps(data))
    print(f"  merged seed{SEED}")
print("done.")
