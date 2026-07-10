"""Continuous-X OWGAP — LIPSCHITZ policy class on the ROBUST methods (IPW-O-X, DR-O-X, Hajek-O-X, DR-O-W).

Same idea as run_owgap_lipschitz.py but for the Γ-robust solvers (which now accept a `lipschitz` kwarg):
at a fixed operating Γ, add the L-Lipschitz policy-class constraint, train on continuous X, deploy on a
test set via KNN, and sweep L (L=∞ = the unit-by-unit baseline). Shows whether the policy class compounds
with the Γ / Wasserstein robustness.

Usage:
  python3 assets/run_owgap_lipschitz_robust.py --out assets/exp_owgap_cont/owgap_lipschitz_robust.json \
      --n 400 --n-test 2000 --seeds 5 --gamma 2 --k 50 --workers 2
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "extensions" / "KNN")); sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
DGP_PATH = str((ROOT / "assets/exp_owgap_cont/dgp.py").resolve())
METHODS = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
LGRID = [None, 20.0, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]
_W = {}

def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)

def _init(dgp_path):
    import common; _W["common"] = common
    from knn import extend_with_knn; _W["knn"] = extend_with_knn
    from shapley import extend_with_shapley; _W["shp"] = extend_with_shapley
    sp = importlib.util.spec_from_file_location("dgpmod", dgp_path); d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d
    S = {}
    S["IPW-O-X"] = _load("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
    S["DoublyRobust-O-X"] = _load("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped")
    S["Hajek-O-X"] = _load("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py", "solve_hajek_o_x_uncapped")
    S["IPW-O-W"] = _load("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
    S["DoublyRobust-O-W"] = _load("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")
    _W["S"] = S

def solve_job(job):
    seed, N, Nte, g, k, ngrid, ceps = job
    common = _W["common"]; d = _W["dgp"]; S = _W["S"]; knn = _W["knn"]; shp = _W["shp"]; K = d.K
    obs, full = d.generate(N, seed); X, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(Nte, seed + 1000); Xte = te["X"]; Y1t, Y0t = ft["Y1"], ft["Y0"]
    Xg = np.linspace(-1.0, 1.0, ngrid).reshape(-1, 1)
    w, _ = common.ipw_weights_from_data(X, T, K); wraw, _ = common.ipw_weights_from_data(X, T, K, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X); eps = tuple(common.tight_epsilon(Dm, T, w, K, is_distance=True, c_eps=ceps))
    Lkeys = ["inf" if L is None else ("%g" % L) for L in LGRID]
    out = {m: {"val": {}, "val_shp": {}, "grid": {}, "grid_shp": {}} for m in METHODS}
    def call(m, L):
        if m == "IPW-O-X": return S[m](X, T, Y, w, n_arms=K, Gamma=g, discretize=False, lipschitz=L)
        if m == "DoublyRobust-O-X": return S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, lipschitz=L)
        if m == "Hajek-O-X": return S[m](X, T, Y, wraw, n_arms=K, Gamma=g, maximize=True, discretize=False, lipschitz=L)
        if m == "IPW-O-W": return S[m](X, T, Y, w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=L)
        return S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=L)
    for m in METHODS:
        for L, lk in zip(LGRID, Lkeys):
            try:
                res = call(m, L)
                pe = knn(Xte, X, res.pi[1], k=k)
                out[m]["val"][lk] = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
                ps = shp(Xte, X, res.pi[1])
                out[m]["val_shp"][lk] = float(np.mean(ps * Y1t + (1 - ps) * Y0t))
                if seed == 0:
                    out[m]["grid"][lk] = [round(float(v), 4) for v in knn(Xg, X, res.pi[1], k=k)]
                    out[m]["grid_shp"][lk] = [round(float(v), 4) for v in shp(Xg, X, res.pi[1])]
            except Exception as ex:
                out[m]["val"][lk] = float("nan"); out[m]["val_shp"][lk] = float("nan")
                print("%s L=%s seed%d FAIL: %s" % (m, lk, seed, ex), flush=True)
    orc = d.oracle_policy(Xte.ravel())
    refs = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)), "never_treat": float(np.mean(Y0t)), "all_treat": float(np.mean(Y1t))}
    return seed, out, refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/exp_owgap_cont/owgap_lipschitz_robust.json")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--n-test", type=int, default=2000, dest="ntest")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--k", type=int, default=50); ap.add_argument("--ngrid", type=int, default=121)
    ap.add_argument("--ceps", type=float, default=1.0); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    seeds = list(range(a.seeds)); Path("gurobi.env").write_text("Threads 1\n")
    Lkeys = ["inf" if L is None else ("%g" % L) for L in LGRID]
    jobs = [(sd, a.n, a.ntest, a.gamma, a.k, a.ngrid, a.ceps) for sd in seeds]
    print("lipschitz-robust: N=%d Nte=%d seeds=%d Gamma=%g k=%d workers=%d" % (a.n, a.ntest, a.seeds, a.gamma, a.k, a.workers), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(DGP_PATH,)) as pool:
        res = pool.map(solve_job, jobs)
    Xg = [round(float(x), 4) for x in np.linspace(-1.0, 1.0, a.ngrid)]
    methods = {}
    for m in METHODS:
        tm = {lk: round(float(np.nanmean([o[m]["val"][lk] for _, o, _ in res])), 4) for lk in Lkeys}
        tsh = {lk: round(float(np.nanmean([o[m]["val_shp"][lk] for _, o, _ in res])), 4) for lk in Lkeys}
        ts = {lk: round(float(np.nanstd([o[m]["val"][lk] for _, o, _ in res])), 4) for lk in Lkeys}
        gp0 = {}; gp0s = {}
        for _, o, _r in res:
            if o[m]["grid"]: gp0 = o[m]["grid"]; gp0s = o[m].get("grid_shp", {}); break
        methods[m] = {"test_mean": tm, "test_shp": tsh, "test_sd": ts, "gridpol": gp0, "gridpol_shp": gp0s}
    def ref(k): return round(float(np.mean([r[k] for _, _, r in res])), 4)
    out = {"N_train": a.n, "N_test": a.ntest, "seeds": seeds, "k": a.k, "gamma": a.gamma, "Lgrid": Lkeys, "grid": Xg,
           "oracle": ref("oracle"), "never_treat": ref("never_treat"), "all_treat": ref("all_treat"), "methods": methods}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("saved %s  (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)
    print("\n=== TEST realized outcome vs L (Γ=%g, mean over %d seeds) — oracle=%.3f never=%.3f ===" % (a.gamma, a.seeds, out["oracle"], out["never_treat"]))
    for ext, key in [("KNN", "test_mean"), ("Shapley", "test_shp")]:
        print("-- %s --  " % ext + "".join("  %7s" % lk for lk in Lkeys))
        for m in METHODS:
            print("%-18s" % m + "".join("  %7.3f" % methods[m][key][lk] for lk in Lkeys))
    print("DONE_MARKER", flush=True)

if __name__ == "__main__":
    main()
