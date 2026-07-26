"""Continuous-X OWGAP — L × Γ 2-D sweep for the ROBUST methods (choose-your-method surface).

For each robust method (IPW-O-X, DoublyRobust-O-X, Hajek-O-X, IPW-O-W, DoublyRobust-O-W) sweep BOTH the
Lipschitz policy class L and the sensitivity Γ, train on continuous X (discretize=False), deploy on a
test set via KNN. Output surface[method][Γ][L] = test realized outcome, so the report can slice it
(E-vs-L at fixed Γ, or E-vs-Γ at fixed L) with a method checklist.

Usage:
  python3 assets/run_owgap_lip_gamma_2d.py --out assets/exp_owgap_cont/owgap_lip_gamma_2d.json \
      --n 400 --n-test 2000 --seeds 3 --k 50 --workers 2
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "extensions" / "KNN")); sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
DGP_PATH = str((ROOT / "assets/exp_owgap_cont/dgp.py").resolve())
GAMMAS = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0]
LGRID = [None, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.5]
METHODS = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
_W = {}

def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)

def _init(dgp_path):
    import common; _W["common"] = common
    from knn import extend_with_knn; _W["knn"] = extend_with_knn
    from shapley import extend_with_shapley, extract_support
    _W["shp"] = extend_with_shapley; _W["extract"] = extract_support
    sp = importlib.util.spec_from_file_location("dgpmod", dgp_path); d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d
    S = {}
    S["IPW-O-X"] = _load("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
    S["DoublyRobust-O-X"] = _load("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped")
    S["Hajek-O-X"] = _load("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py", "solve_hajek_o_x_uncapped")
    S["IPW-O-W"] = _load("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
    S["DoublyRobust-O-W"] = _load("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")
    _W["S"] = S

PGRID = np.linspace(-1.0, 1.0, 41)   # fixed eval grid for saved policy curves (seed 0)

def _shapley_fast(Xnew, sX, sp, block=200):
    """Vectorized closed-form Shapley operator pi^S(x) = min_k max_j A_jk(x) (extensions/Shapley),
    chunked over test points. Numerically identical to shapley.extend_with_shapley (~50x faster)."""
    if Xnew.ndim == 1: Xnew = Xnew.reshape(-1, 1)
    out = np.empty(Xnew.shape[0])
    diag = np.arange(sX.shape[0])
    for s0 in range(0, Xnew.shape[0], block):
        xb = Xnew[s0:s0 + block]                                        # (b, d)
        d = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))   # (b, K) distances
        S = d[:, :, None] + d[:, None, :]                               # (b, K, K)  d_j + d_k
        S = np.where(S > 0, S, 1.0)
        A = (d[:, None, :] * sp[None, :, None] + d[:, :, None] * sp[None, None, :]) / S   # A[b,j,k]
        A[:, diag, diag] = sp[None, :]
        val = A.max(axis=1).min(axis=1)                                 # min_k max_j
        ex = d.min(axis=1) <= 0.0                                       # exact-match fast path
        if ex.any(): val[ex] = sp[d[ex].argmin(axis=1)]
        out[s0:s0 + block] = val
    return out

def solve_job(job):
    seed, N, Nte, k, ceps, deploy = job
    common = _W["common"]; d = _W["dgp"]; S = _W["S"]; knn = _W["knn"]; K = d.K
    if deploy == "shapley":
        xsup = _W["extract"]
        def dep(Xnew, Xs, vals):
            sX, sp = xsup(Xs, vals)
            return _shapley_fast(np.asarray(Xnew, float), np.asarray(sX, float), np.asarray(sp, float).ravel())
    else:
        def dep(Xnew, Xs, vals): return knn(Xnew, Xs, vals, k=k)
    obs, full = d.generate(N, seed); X, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(Nte, seed + 1000); Xte = te["X"]; Y1t, Y0t = ft["Y1"], ft["Y0"]
    w, _ = common.ipw_weights_from_data(X, T, K); wraw, _ = common.ipw_weights_from_data(X, T, K, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X); eps = tuple(common.tight_epsilon(Dm, T, w, K, is_distance=True, c_eps=ceps))
    Gk = ["%g" % g for g in GAMMAS]; Lkeys = ["inf" if L is None else ("%g" % L) for L in LGRID]
    grid = {m: {gk: {lk: float("nan") for lk in Lkeys} for gk in Gk} for m in METHODS}
    # SAVE EVERYTHING, EVERY SEED: deployed curves AND raw per-unit support policies.
    # Re-solving the LPs because an output was discarded is never acceptable.
    pol = {m: {gk: {} for gk in Gk} for m in METHODS}
    sup = {m: {gk: {} for gk in Gk} for m in METHODS}
    Pg = PGRID.reshape(-1, 1)
    def call(m, g, L):
        if m == "IPW-O-X": return S[m](X, T, Y, w, n_arms=K, Gamma=g, discretize=False, lipschitz=L)
        if m == "DoublyRobust-O-X": return S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, lipschitz=L)
        if m == "Hajek-O-X": return S[m](X, T, Y, wraw, n_arms=K, Gamma=g, maximize=True, discretize=False, lipschitz=L)
        if m == "IPW-O-W": return S[m](X, T, Y, w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=L)
        return S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=L)
    for m in METHODS:
        for g in GAMMAS:
            for L, lk in zip(LGRID, Lkeys):
                try:
                    res = call(m, g, L); pe = dep(Xte, X, res.pi[1])
                    grid[m]["%g" % g][lk] = float(np.mean(pe * Y1t + (1 - pe) * Y0t))
                    pol[m]["%g" % g][lk] = [round(float(v), 4) for v in dep(Pg, X, res.pi[1])]
                    sup[m]["%g" % g][lk] = [round(float(v), 4) for v in res.pi[1]]
                except Exception as ex:
                    print("%s g%s L%s seed%d FAIL: %s" % (m, g, lk, seed, ex), flush=True)
    orc = d.oracle_policy(Xte.ravel())
    # naive DoublyRobust-X-X baseline: per-unit sign of the cross-fit linear CATE-hat, deployed off-support
    nv = (mu[:, 1] - mu[:, 0] > 0).astype(float)
    nve = dep(Xte, X, nv)
    refs = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)), "never_treat": float(np.mean(Y0t)),
            "all_treat": float(np.mean(Y1t)),
            "naive_dr": float(np.mean(nve * Y1t + (1 - nve) * Y0t))}
    pol["_refs"] = {"oracle": [round(float(v), 4) for v in d.oracle_policy(PGRID)],
                    "naive_dr": [round(float(v), 4) for v in dep(Pg, X, nv)]}
    sup["_X"] = [round(float(v), 5) for v in np.asarray(X).ravel()]
    sup["_naive_dr"] = [round(float(v), 4) for v in nv]
    return seed, grid, refs, pol, sup

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/exp_owgap_cont/owgap_lip_gamma_2d.json")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--n-test", type=int, default=2000, dest="ntest")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--ceps", type=float, default=1.0); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--dgp", default=DGP_PATH, help="path to a DGP module (default: exp_owgap_cont)")
    ap.add_argument("--deploy", default="knn", choices=["knn", "shapley"],
                    help="off-support deployment: knn (k-NN average, smooths) or shapley "
                         "(closed-form Lipschitz min-max interpolant, exact at support points)")
    a = ap.parse_args()
    seeds = list(range(a.seeds)); Path("gurobi.env").write_text("Threads 1\n")
    Gk = ["%g" % g for g in GAMMAS]; Lkeys = ["inf" if L is None else ("%g" % L) for L in LGRID]
    jobs = [(sd, a.n, a.ntest, a.k, a.ceps, a.deploy) for sd in seeds]
    print("L×Γ 2-D (%d methods): N=%d Nte=%d seeds=%d gammas=%s Ls=%s workers=%d deploy=%s dgp=%s" % (len(METHODS), a.n, a.ntest, a.seeds, Gk, Lkeys, a.workers, a.deploy, a.dgp), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(str(Path(a.dgp).resolve()),)) as pool:
        res = pool.map(solve_job, jobs)
    surface = {m: {gk: {lk: round(float(np.nanmean([o[m][gk][lk] for _, o, _, _, _ in res])), 4) for lk in Lkeys} for gk in Gk} for m in METHODS}
    oracle = round(float(np.mean([r["oracle"] for _, _, r, _, _ in res])), 4)
    never = round(float(np.mean([r["never_treat"] for _, _, r, _, _ in res])), 4)
    all_treat = round(float(np.mean([r["all_treat"] for _, _, r, _, _ in res])), 4)
    naive_dr = round(float(np.mean([r["naive_dr"] for _, _, r, _, _ in res])), 4)
    pol_by_seed = {str(sd): p for sd, _, _, p, _ in res}
    sup_by_seed = {str(sd): s for sd, _, _, _, s in res}
    refs_by_seed = {str(sd): r for sd, _, r, _, _ in res}
    policies = pol_by_seed.get("0")
    support_pol = sup_by_seed.get("0")
    best = {}
    for m in METHODS:
        cells = [(gk, lk, surface[m][gk][lk]) for gk in Gk for lk in Lkeys if surface[m][gk][lk] == surface[m][gk][lk]]
        bg, bl, bv = max(cells, key=lambda t: t[2]) if cells else ("", "", float("nan"))
        best[m] = {"gamma": bg, "L": bl, "value": bv}
    bm = max(METHODS, key=lambda m: best[m]["value"])
    out = {"N_train": a.n, "N_test": a.ntest, "seeds": seeds, "k": a.k, "deploy": a.deploy,
           "methods": METHODS, "gammas": Gk, "Lgrid": Lkeys,
           "oracle": oracle, "never_treat": never, "all_treat": all_treat, "naive_dr": naive_dr,
           "dgp": str(Path(a.dgp).resolve()), "policy_grid": PGRID.tolist(), "policies_seed0": policies,
           "policies_support_seed0": support_pol,
           "policies_by_seed": pol_by_seed, "policies_support_by_seed": sup_by_seed,
           "refs_by_seed": refs_by_seed,
           "surface": surface, "best": best,
           "best_overall": {"method": bm, "gamma": best[bm]["gamma"], "L": best[bm]["L"], "value": best[bm]["value"]}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("saved %s  (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)
    print("\n=== best per method (test E[Y], %d seeds) — oracle=%.3f never=%.3f all=%.3f naiveDR=%.3f ===" % (a.seeds, oracle, never, all_treat, naive_dr))
    for m in METHODS:
        print("%-18s best %.3f at Γ=%s, L=%s" % (m, best[m]["value"], best[m]["gamma"], best[m]["L"]))
    print("BEST OVERALL: %s Γ=%s L=%s -> %.3f" % (bm, best[bm]["gamma"], best[bm]["L"], best[bm]["value"]))
    # heatmap for the best method
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        M = np.array([[surface[bm][gk][lk] for lk in Lkeys] for gk in Gk], float)
        fig, ax = plt.subplots(figsize=(8.2, 4.7))
        im = ax.imshow(M, aspect="auto", cmap="viridis", origin="upper", vmin=max(never, -0.1), vmax=oracle)
        ax.set_xticks(range(len(Lkeys))); ax.set_xticklabels(Lkeys); ax.set_yticks(range(len(Gk))); ax.set_yticklabels(["Γ=" + g for g in Gk])
        ax.set_xlabel("Lipschitz L  (inf = unit-by-unit)"); ax.set_ylabel("Γ (sensitivity)")
        for i in range(len(Gk)):
            for j in range(len(Lkeys)):
                ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=7, color="white" if M[i, j] < (never + oracle) / 2 else "black")
        fig.colorbar(im, ax=ax, label="test realised E[Y] (oracle=%.2f)" % oracle)
        ax.set_title("%s: test realized outcome over L × Γ (best %.3f)" % (bm, best[bm]["value"]), fontsize=9.5)
        fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(a.out), "lip_gamma_2d.png"), dpi=120); plt.close(fig)
        print("saved lip_gamma_2d.png", flush=True)
    except Exception as ex:
        print("PNG skipped: %s" % ex, flush=True)
    print("DONE_MARKER", flush=True)

if __name__ == "__main__":
    main()
