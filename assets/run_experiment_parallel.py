"""Reusable PARALLEL experiment runner. Runs the 8-method policy-optimization sweep over (regime, seed) jobs
concurrently with multiprocessing — point it at any DGP module and scale by --workers on a big server.

The DGP module (a .py file) must expose:
  LEVELS (1-D np array of the discrete X grid), CAP (tuple, e.g. (1.0,0.5)), K (=2),
  generate(n, seed) -> (observed{X,T,Y}, full), grid_truth(), exact_value(pi_grid) -> float, oracle_policy(X) (optional).

Output JSON matches what the build_* scripts expect:
  {N_train, seeds, gammas, oracle, grid, regimes:{reg:{mean,sd,policy_seed0,policy_by_seed}}}

Usage:
  python3 assets/run_experiment_parallel.py --dgp assets/exp_owwin3/dgp.py --out assets/exp_owwin3/owwin3_results.json \
      --n 700 --seeds 5 --gammas 1,1.5,2,3,4,6,8 --regimes uncap,cap --ceps 1.0 --workers 8 --threads 1
On a big server just raise --workers (one core per (regime,seed) job) and give --threads 2-4 each if cores allow.
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]; OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]; OW = ["IPW-O-W", "DoublyRobust-O-W"]; ALL = XX + OX + OW
_W = {}   # per-worker cache: common, dgp, solvers

def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)

def _init(dgp_path):
    import common; _W["common"] = common
    sp = importlib.util.spec_from_file_location("dgpmod", dgp_path); d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d
    S = {}
    for reg, suf, Cap in [("uncap", "uncapped", "Uncapped"), ("cap", "capped", "Capped")]:
        S[(reg, "IPW-X-X")] = _load("methods/IPW-X-X/%s/ipw_x_x_%s.py" % (Cap, suf), "solve_ipw_x_x_%s" % suf)
        S[(reg, "DoublyRobust-X-X")] = _load("methods/DoublyRobust-X-X/%s/doublyrobust_x_x_%s.py" % (Cap, suf), "solve_doublyrobust_x_x_%s" % suf)
        S[(reg, "Direct-X-X")] = _load("methods/Direct-X-X/%s/direct_x_x_%s.py" % (Cap, suf), "solve_direct_x_x_%s" % suf)
        S[(reg, "IPW-O-X")] = _load("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf)
        S[(reg, "DoublyRobust-O-X")] = _load("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf)
        S[(reg, "Hajek-O-X")] = _load("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (Cap, suf), "solve_hajek_o_x_%s" % suf)
        S[(reg, "IPW-O-W")] = _load("methods/IPW-O-W/%s/ipw_o_w_%s.py" % (Cap, suf), "solve_ipw_o_w_%s" % suf)
        S[(reg, "DoublyRobust-O-W")] = _load("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf)
    _W["S"] = S

def _gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
def _tg(res, LV):
    s = res.support_X.ravel(); lv = np.array(sorted(set(np.round(s, 6))))
    pol = np.array([float(res.pi[1, np.where(np.round(s, 6) == round(float(c), 6))[0][0]]) for c in lv])
    idx = {round(float(c), 6): i for i, c in enumerate(lv)}
    return np.array([pol[idx[round(float(v), 6)]] for v in LV])

def solve_job(job):
    """One (regime, seed): returns dict method-> {'val':[by gamma], 'pol':{gk:grid}}."""
    reg, seed, N, gammas, ceps = job
    common = _W["common"]; d = _W["dgp"]; S = _W["S"]; LV = d.LEVELS; K = d.K
    kw0 = {} if reg == "uncap" else {"cap": d.CAP}
    obs, _ = d.generate(N, seed)
    w, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K)
    wraw, _ = common.ipw_weights_from_data(obs["X"], obs["T"], K, normalize=False)
    mu = common.outcome_means(obs["X"], obs["T"], obs["Y"], n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(obs["X"]); eps = tuple(common.tight_epsilon(Dm, obs["T"], w, K, is_distance=True, c_eps=ceps))
    out = {m: {"val": [], "pol": {}} for m in ALL}
    # XX (Γ-free)
    flat = {"IPW-X-X": S[(reg, "IPW-X-X")](obs["X"], obs["T"], obs["Y"], w, n_arms=K, discretize=False, **kw0),
            "DoublyRobust-X-X": S[(reg, "DoublyRobust-X-X")](obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, discretize=False, **kw0),
            "Direct-X-X": S[(reg, "Direct-X-X")](obs["X"], obs["T"], obs["Y"], n_arms=K, discretize=False, **kw0)}
    for m, res in flat.items():
        v = d.exact_value(_tg(res, LV)); g0 = [round(float(x), 4) for x in _tg(res, LV)]
        out[m]["val"] = [round(v, 4)] * len(gammas)
        for g in gammas: out[m]["pol"][_gk(g)] = g0
    for g in gammas:
        res = {"IPW-O-X": S[(reg, "IPW-O-X")](obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, discretize=False, **kw0),
               "DoublyRobust-O-X": S[(reg, "DoublyRobust-O-X")](obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, **kw0),
               "Hajek-O-X": S[(reg, "Hajek-O-X")](obs["X"], obs["T"], obs["Y"], wraw, n_arms=K, Gamma=g, maximize=True, discretize=False, **kw0),
               "IPW-O-W": S[(reg, "IPW-O-W")](obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0),
               "DoublyRobust-O-W": S[(reg, "DoublyRobust-O-W")](obs["X"], obs["T"], obs["Y"], w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, **kw0)}
        for m in OX + OW:
            out[m]["val"].append(round(d.exact_value(_tg(res[m], LV)), 4))
            out[m]["pol"][_gk(g)] = [round(float(x), 4) for x in _tg(res[m], LV)]
        try:
            with open("assets/exp_owgap/.gamma_progress", "a") as _pf:
                _pf.write("%s %s %s %s\n" % (ceps, reg, seed, g))
        except Exception:
            pass
    return reg, str(seed), out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dgp", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=700); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gammas", default="1,1.5,2,3,4,6,8"); ap.add_argument("--regimes", default="uncap,cap")
    ap.add_argument("--ceps", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--threads", type=int, default=1, help="Gurobi threads per worker (workers*threads <= cores)")
    a = ap.parse_args()
    gammas = [float(x) for x in a.gammas.split(",")]; regimes = a.regimes.split(","); seeds = list(range(a.seeds))
    # cap Gurobi threads per process via gurobi.env in CWD (read at env creation)
    Path("gurobi.env").write_text("Threads %d\n" % a.threads)
    dgp_path = str((ROOT / a.dgp).resolve()) if not os.path.isabs(a.dgp) else a.dgp
    sp = importlib.util.spec_from_file_location("dgpmain", dgp_path); d0 = importlib.util.module_from_spec(sp); sp.loader.exec_module(d0)
    orc = d0.exact_value(d0.grid_truth()["oracle"]); LV = d0.LEVELS
    jobs = [(reg, sd, a.n, gammas, a.ceps) for reg in regimes for sd in seeds]
    print("parallel run: %d jobs (%s × %d seeds), workers=%d, threads/worker=%d, Γ=%s" % (len(jobs), regimes, a.seeds, a.workers, a.threads, gammas), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(dgp_path,)) as pool:
        res = pool.map(solve_job, jobs)
    # aggregate
    out = {"N_train": a.n, "seeds": seeds, "gammas": gammas, "oracle": round(orc, 4), "grid": [float(v) for v in LV], "regimes": {}}
    for reg in regimes:
        per = {m: {} for m in ALL}                       # method -> seedkey -> {val, pol}
        for r, sk, o in res:
            if r == reg:
                for m in ALL: per[m][sk] = o[m]
        mean = {m: [round(float(np.mean([per[m][str(s)]["val"][gi] for s in seeds])), 4) for gi in range(len(gammas))] for m in ALL}
        sd_ = {m: [round(float(np.std([per[m][str(s)]["val"][gi] for s in seeds])), 4) for gi in range(len(gammas))] for m in ALL}
        polS = {m: {str(s): per[m][str(s)]["pol"] for s in seeds} for m in ALL}
        for m in ALL:
            polS[m]["avg"] = {_gk(g): [round(float(np.mean([polS[m][str(s)][_gk(g)][j] for s in seeds])), 4) for j in range(len(LV))] for g in gammas}
        out["regimes"][reg] = {"mean": mean, "sd": sd_, "policy_seed0": {m: polS[m]["0"] for m in ALL}, "policy_by_seed": polS}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("saved %s  oracle=%.3f  (%.1f min, %d jobs on %d workers)" % (a.out, orc, (time.time() - t0) / 60, len(jobs), a.workers), flush=True)
    for reg in regimes:
        M = out["regimes"][reg]["mean"]; print("[%s] DR-O-W=%s  DR-O-X=%s" % (reg, M["DoublyRobust-O-W"], M["DoublyRobust-O-X"]), flush=True)

if __name__ == "__main__":
    main()
