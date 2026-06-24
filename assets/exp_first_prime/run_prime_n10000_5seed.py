"""5-seed first-prime N_train=10000 study, XX + OX methods, all Γ in {1,2,3,4,6,7.46,8,20}, both regimes.
Realised test E[Y] on te.Ypot (same ruler as before); oracle = best-means policy realised on test. Saves
prime_n10000_5seed_results.json with per-seed values + 5-seed mean/SD per method per Γ per regime."""
import sys, json, importlib.util, time
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
SOLV = {"uncap": {"IPW-X-X": Lf("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped"),
                  "DoublyRobust-X-X": Lf("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped"),
                  "Direct-X-X": Lf("methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py", "solve_direct_x_x_uncapped"),
                  "IPW-O-X": Lf("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped"),
                  "DoublyRobust-O-X": Lf("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped"),
                  "Hajek-O-X": Lf("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py", "solve_hajek_o_x_uncapped")},
        "cap": {"IPW-X-X": Lf("methods/IPW-X-X/Capped/ipw_x_x_capped.py", "solve_ipw_x_x_capped"),
                "DoublyRobust-X-X": Lf("methods/DoublyRobust-X-X/Capped/doublyrobust_x_x_capped.py", "solve_doublyrobust_x_x_capped"),
                "Direct-X-X": Lf("methods/Direct-X-X/Capped/direct_x_x_capped.py", "solve_direct_x_x_capped"),
                "IPW-O-X": Lf("methods/IPW-O-X/Capped/ipw_o_x_capped.py", "solve_ipw_o_x_capped"),
                "DoublyRobust-O-X": Lf("methods/DoublyRobust-O-X/Capped/doublyrobust_o_x_capped.py", "solve_doublyrobust_o_x_capped"),
                "Hajek-O-X": Lf("methods/Hajek-O-X/Capped/hajek_o_x_capped.py", "solve_hajek_o_x_capped")}}
K = dgp.K; NTR, NTE = 10000, 20000; SEEDS = [0, 1, 2, 3, 4]
GAMMAS = [1.0, 2.0, 3.0, 4.0, 6.0, 7.46, 8.0, 20.0]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]; OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]

def constrained_oracle(X, V, cap):
    Xr = np.round(np.asarray(X).ravel(), 6); lv = np.array(sorted(set(Xr))); idx = {round(float(c), 6): s for s, c in enumerate(lv)}; ns = len(lv); n = len(Xr)
    cnt = np.zeros(ns); Vs = np.zeros((ns, K))
    for j, x in enumerate(Xr): cnt[idx[round(float(x), 6)]] += 1; Vs[idx[round(float(x), 6)]] += V[j]
    m = gp.Model(); m.Params.OutputFlag = 0; pi = m.addVars(ns, K, lb=0, ub=1)
    for s in range(ns): m.addConstr(gp.quicksum(pi[s, k] for k in range(K)) == 1)
    for k in range(K): m.addConstr(gp.quicksum(cnt[s] * pi[s, k] for s in range(ns)) <= cap[k] * n)
    m.setObjective(gp.quicksum(Vs[s, k] * pi[s, k] for s in range(ns) for k in range(K)), GRB.MAXIMIZE); m.optimize()
    P = np.array([[pi[s, k].X for k in range(K)] for s in range(ns)]); return np.array([P[idx[round(float(x), 6)]] for x in Xr]).T

# per-seed: regimes -> {method -> [val per Γ]}, plus oracle per regime
PERSEED = {rg: {m: [] for m in XX + OX} for rg in ("uncap", "cap")}
ORACLE = {"uncap": [], "cap": []}
t_all = time.time()
for s in SEEDS:
    rng = np.random.default_rng(s); tr = dgp.generate(NTR, rng); te = dgp.generate(NTE, rng)
    w, _ = common.ipw_weights_from_data(tr.X, tr.T, K); muhat = common.outcome_means(tr.X, tr.T, tr.Y, n_arms=K, cross_fit=True)
    by = {}
    for j, xv in enumerate(np.round(tr.X.ravel(), 6)): by.setdefault(round(float(xv), 6), j)
    nn = np.array([by[round(float(x), 6)] for x in np.round(te.X.ravel(), 6)])
    def val(pi): pi = np.asarray(pi, float); pte = pi[:, nn]; return float((pte * te.Ypot.T).sum() / pte.shape[1])
    for rg, cap in (("uncap", dgp.CAP_UNCAP), ("cap", dgp.CAP_CAP)):
        S = SOLV[rg]; kw = {} if rg == "uncap" else {"cap": cap}
        bm = constrained_oracle(te.X, te.mu, cap); ORACLE[rg].append(round(float((bm * te.Ypot.T).sum() / bm.shape[1]), 4))
        vx = val(S["IPW-X-X"](tr.X, tr.T, tr.Y, w, n_arms=K, discretize=False, **kw).pi)
        vd = val(S["DoublyRobust-X-X"](tr.X, tr.T, tr.Y, w, muhat, n_arms=K, discretize=False, **kw).pi)
        vr = val(S["Direct-X-X"](tr.X, tr.T, tr.Y, n_arms=K, discretize=False, **kw).pi)
        for m, v in (("IPW-X-X", vx), ("DoublyRobust-X-X", vd), ("Direct-X-X", vr)):
            PERSEED[rg][m].append([round(v, 4)] * len(GAMMAS))
        for m in OX: PERSEED[rg][m].append([])
        for g in GAMMAS:
            ro = S["IPW-O-X"](tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(g), discretize=False, **kw)
            rd = S["DoublyRobust-O-X"](tr.X, tr.T, tr.Y, w, muhat, n_arms=K, Gamma=float(g), discretize=False, **kw)
            rh = S["Hajek-O-X"](tr.X, tr.T, tr.Y, w, n_arms=K, Gamma=float(g), maximize=True, discretize=False, **kw)
            PERSEED[rg]["IPW-O-X"][-1].append(round(val(ro.pi), 4))
            PERSEED[rg]["DoublyRobust-O-X"][-1].append(round(val(rd.pi), 4))
            PERSEED[rg]["Hajek-O-X"][-1].append(round(val(rh.pi), 4))
    print("seed %d done (%.1f min elapsed)" % (s, (time.time() - t_all) / 60), flush=True)

agg = {"N_train": NTR, "N_test": NTE, "seeds": SEEDS, "gammas": GAMMAS, "regimes": {}}
for rg in ("uncap", "cap"):
    R = {"mean": {}, "sd": {}, "oracle_mean": round(float(np.mean(ORACLE[rg])), 4), "oracle_sd": round(float(np.std(ORACLE[rg])), 4)}
    for m in XX + OX:
        A = np.array(PERSEED[rg][m], float)  # (seeds, Γ)
        R["mean"][m] = [round(float(v), 4) for v in A.mean(0)]; R["sd"][m] = [round(float(v), 4) for v in A.std(0)]
    R["per_seed"] = PERSEED[rg]; R["oracle_per_seed"] = ORACLE[rg]
    agg["regimes"][rg] = R
(HERE / "prime_n10000_5seed_results.json").write_text(json.dumps(agg, indent=2))
print("saved prime_n10000_5seed_results.json (%.1f min total)" % ((time.time() - t_all) / 60), flush=True)
for rg in ("uncap", "cap"):
    R = agg["regimes"][rg]; print("[%s] oracle=%.4f+-%.4f  DR-O-X mean=%s" % (rg, R["oracle_mean"], R["oracle_sd"], R["mean"]["DoublyRobust-O-X"]), flush=True)
