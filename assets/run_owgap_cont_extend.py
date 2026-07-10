"""Continuous-X OWGAP (no mesh) — policy EXTENSION (KNN + Shapley) + TEST-set realized outcome.

Raw per-unit training (**dkw) gives a policy defined only at the training points. To deploy
it at new X (a held-out test set, or a dense grid for plotting) we lift it off-support with the two
extension operators in extensions/{KNN,Shapley}. Neither needs a grid/mesh: they evaluate at ANY x
directly from the raw support. We then score each extended policy by its realized outcome on an
INDEPENDENT test set:

    V(pi_ext) = (1/m) Σ_j [ pi_ext_1(Xte_j)·Y1te_j + pi_ext_0(Xte_j)·Y0te_j ]

7 methods (IPW-O-W excluded), uncapped, over a Γ sweep. Also emits per-seed extended policy curves on a
dense x-grid (plotting + the interactive policy explorer).

Usage:
  python3 assets/run_owgap_cont_extend.py --out assets/exp_owgap_cont/owgap_cont_extend.json \
      --n 400 --n-test 2000 --seeds 5 --gammas 1,1.5,2,3,4,6,8 --k 50 --workers 2 --threads 1
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
sys.path.insert(0, str(ROOT / "extensions" / "KNN"))
DGP_PATH = str((ROOT / "assets/exp_owgap_cont/dgp.py").resolve())

XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["DoublyRobust-O-W"]                        # IPW-O-W intentionally EXCLUDED
ALLM = XX + OX + OW
EXTS = ["knn", "shp"]
_W = {}

def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)

def _gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)

def _init(dgp_path):
    import common; _W["common"] = common
    from knn import extend_with_knn_multiarm
    from shapley import extend_with_shapley_multiarm
    _W["ext"] = {"knn": lambda Xq, Xtr, pi: extend_with_knn_multiarm(Xq, Xtr, pi, k=_W["k"]),
                 "shp": lambda Xq, Xtr, pi: extend_with_shapley_multiarm(Xq, Xtr, pi)}
    sp = importlib.util.spec_from_file_location("dgpmod", dgp_path); d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d
    S = {}
    suf, Cap = "uncapped", "Uncapped"
    S["IPW-X-X"] = _load("methods/IPW-X-X/%s/ipw_x_x_%s.py" % (Cap, suf), "solve_ipw_x_x_%s" % suf)
    S["DoublyRobust-X-X"] = _load("methods/DoublyRobust-X-X/%s/doublyrobust_x_x_%s.py" % (Cap, suf), "solve_doublyrobust_x_x_%s" % suf)
    S["Direct-X-X"] = _load("methods/Direct-X-X/%s/direct_x_x_%s.py" % (Cap, suf), "solve_direct_x_x_%s" % suf)
    S["IPW-O-X"] = _load("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf)
    S["DoublyRobust-O-X"] = _load("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf)
    S["Hajek-O-X"] = _load("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (Cap, suf), "solve_hajek_o_x_%s" % suf)
    S["DoublyRobust-O-W"] = _load("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf)
    _W["S"] = S

def solve_job(job):
    seed, N, Nte, gammas, ceps, k, ngrid, mesh = job
    _W["k"] = k
    dkw = {"discretize": True, "mesh": int(mesh), "mesh_range": (-1.0, 1.0)} if mesh and int(mesh) > 0 else {"discretize": False}
    common = _W["common"]; d = _W["dgp"]; S = _W["S"]; K = d.K
    if "ext" not in _W:                                     # (re)build ext closures with this k
        from knn import extend_with_knn_multiarm
        from shapley import extend_with_shapley_multiarm
        _W["ext"] = {"knn": lambda Xq, Xtr, pi: extend_with_knn_multiarm(Xq, Xtr, pi, k=_W["k"]),
                     "shp": lambda Xq, Xtr, pi: extend_with_shapley_multiarm(Xq, Xtr, pi)}
    EXT = _W["ext"]
    obs, full = d.generate(N, seed)
    Xtr, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(Nte, seed + 1000)
    Xte = te["X"]; Y1te, Y0te, mu1te, mu0te = ft["Y1"], ft["Y0"], ft["mu1"], ft["mu0"]
    Xg = np.linspace(-1.0, 1.0, ngrid).reshape(-1, 1)
    w, _ = common.ipw_weights_from_data(Xtr, T, K)
    wraw, _ = common.ipw_weights_from_data(Xtr, T, K, normalize=False)
    mu = common.outcome_means(Xtr, T, Y, n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(Xtr)
    eps = tuple(common.tight_epsilon(Dm, T, w, K, is_distance=True, c_eps=ceps))
    ng = len(gammas)
    out = {m: {e: {"val": [np.nan]*ng, "val_nf": [np.nan]*ng, "grid": [None]*ng} for e in EXTS} for m in ALLM}

    def _extend_and_store(m, gi, res, gi_all=None):
        """Extend res.pi to test + grid for both extensions; store at gamma index gi (or all of gi_all)."""
        gis = gi_all if gi_all is not None else [gi]
        for e in EXTS:
            pe = EXT[e](Xte, Xtr, res.pi)                  # (K, m_test)
            v = float(np.mean(pe[1]*Y1te + pe[0]*Y0te))
            vnf = float(np.mean(pe[1]*mu1te + pe[0]*mu0te))
            pg = np.clip(EXT[e](Xg, Xtr, res.pi)[1], 0.0, 1.0)   # treat-prob curve on Xg
            gl = [round(float(x), 4) for x in pg]
            for g in gis:
                out[m][e]["val"][g] = v; out[m][e]["val_nf"][g] = vnf; out[m][e]["grid"][g] = gl

    # --- Gamma-free (X-X): solve+extend once, replicate across Gamma ---
    allg = list(range(ng))
    try: _extend_and_store("IPW-X-X", 0, S["IPW-X-X"](Xtr, T, Y, w, n_arms=K, **dkw), gi_all=allg)
    except Exception as ex: print("IPW-X-X seed%d FAIL: %s" % (seed, ex), flush=True)
    try: _extend_and_store("DoublyRobust-X-X", 0, S["DoublyRobust-X-X"](Xtr, T, Y, w, mu, n_arms=K, **dkw), gi_all=allg)
    except Exception as ex: print("DoublyRobust-X-X seed%d FAIL: %s" % (seed, ex), flush=True)
    try: _extend_and_store("Direct-X-X", 0, S["Direct-X-X"](Xtr, T, Y, n_arms=K, **dkw), gi_all=allg)
    except Exception as ex: print("Direct-X-X seed%d FAIL: %s" % (seed, ex), flush=True)

    # --- Gamma-dependent ---
    for gi, g in enumerate(gammas):
        for m, call in [
            ("IPW-O-X", lambda: S["IPW-O-X"](Xtr, T, Y, w, n_arms=K, Gamma=g, **dkw)),
            ("DoublyRobust-O-X", lambda: S["DoublyRobust-O-X"](Xtr, T, Y, w, mu, n_arms=K, Gamma=g, **dkw)),
            ("Hajek-O-X", lambda: S["Hajek-O-X"](Xtr, T, Y, wraw, n_arms=K, Gamma=g, maximize=True, **dkw)),
            ("DoublyRobust-O-W", lambda: S["DoublyRobust-O-W"](Xtr, T, Y, w, mu, n_arms=K, Gamma=g, **dkw, zscore=False, epsilon=eps)),
        ]:
            try: _extend_and_store(m, gi, call())
            except Exception as ex: print("%s seed%d g%s FAIL: %s" % (m, seed, g, ex), flush=True)

    orc = d.oracle_policy(np.asarray(Xte).ravel())
    refs = {"oracle": float(np.mean(orc*Y1te + (1-orc)*Y0te)),
            "oracle_nf": float(np.mean(orc*mu1te + (1-orc)*mu0te)),
            "never_treat": float(np.mean(Y0te)), "all_treat": float(np.mean(Y1te))}
    return seed, out, refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/exp_owgap_cont/owgap_cont_extend.json")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--n-test", type=int, default=2000, dest="ntest")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--gammas", default="1,1.5,2,3,4,6,8")
    ap.add_argument("--ceps", type=float, default=1.0); ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--ngrid", type=int, default=121); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mesh", type=int, default=0, help="0 = raw per-unit training; >0 = train with X snapped to this many levels, then extend")
    a = ap.parse_args()
    gammas = [float(x) for x in a.gammas.split(",")]; seeds = list(range(a.seeds))
    Path("gurobi.env").write_text("Threads %d\n" % a.threads)
    jobs = [(sd, a.n, a.ntest, gammas, a.ceps, a.k, a.ngrid, a.mesh) for sd in seeds]
    tmode = ("mesh=%d" % a.mesh) if a.mesh > 0 else "raw per-unit"
    print("cont-extend: N_train=%d N_test=%d %d seeds gammas=%s k=%d grid=%d workers=%d (uncap, train=%s, then KNN+Shapley extend)"
          % (a.n, a.ntest, a.seeds, gammas, a.k, a.ngrid, a.workers, tmode), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(DGP_PATH,)) as pool:
        res = pool.map(solve_job, jobs)
    _W["k"] = a.k
    Xg = [round(float(x), 4) for x in np.linspace(-1.0, 1.0, a.ngrid)]

    def agg(vals):
        arr = np.array(vals, float)
        return [round(float(np.nanmean(arr[:, gi])), 4) for gi in range(len(gammas))], \
               [round(float(np.nanstd(arr[:, gi])), 4) for gi in range(len(gammas))]

    methods = {}
    for m in ALLM:
        methods[m] = {}
        for e in EXTS:
            vmean, vsd = agg([o[m][e]["val"] for _, o, _ in res])
            vnf, _ = agg([o[m][e]["val_nf"] for _, o, _ in res])
            # mean-over-seeds grid policy per gamma (skip Nones)
            gmean = []
            for gi in range(len(gammas)):
                curves = [o[m][e]["grid"][gi] for _, o, _ in res if o[m][e]["grid"][gi] is not None]
                gmean.append([round(float(v), 4) for v in np.mean(np.array(curves, float), axis=0)] if curves else None)
            methods[m][e] = {"mean": vmean, "sd": vsd, "mean_nf": vnf, "grid_mean": gmean}
    def refagg(k):
        v = np.array([r[k] for _, _, r in res], float)
        return {"mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v)), 4)}
    out = {"N_train": a.n, "N_test": a.ntest, "seeds": seeds, "gammas": gammas, "k": a.k,
           "regime": "uncap", "discretize": bool(a.mesh > 0), "mesh": a.mesh, "train_mode": tmode,
           "x_dist": "uniform(-1,1)", "grid": Xg,
           "oracle": refagg("oracle"), "oracle_nf": refagg("oracle_nf"),
           "never_treat": refagg("never_treat"), "all_treat": refagg("all_treat"), "methods": methods}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("saved %s  (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)

    # --- policy-explorer JSONs (per-seed grid policies), one per extension; named from the out stem ---
    outdir = os.path.dirname(a.out); stem = os.path.splitext(os.path.basename(a.out))[0]
    for e, tag in [("knn", "knn"), ("shp", "shapley")]:
        pol = {}
        for seed, o, _ in res:
            for m in ALLM:
                for gi, g in enumerate(gammas):
                    gl = o[m][e]["grid"][gi]
                    if gl is None: continue
                    pol.setdefault(m, {}).setdefault(str(seed), {})[_gk(g)] = gl
        exp = {"grid": Xg, "gammas": gammas, "regimes": {"uncap": {"policy_by_seed": pol}}}
        p = os.path.join(outdir, "%s_%s_policies.json" % (stem, tag))
        Path(p).write_text(json.dumps(exp)); print("saved %s" % p, flush=True)

    # --- table ---
    print("\n=== TEST realized outcome (mean over %d seeds) — oracle=%.3f never=%.3f all=%.3f ===" %
          (a.seeds, out["oracle"]["mean"], out["never_treat"]["mean"], out["all_treat"]["mean"]))
    for e in EXTS:
        print("-- %s --  " % ("KNN" if e == "knn" else "Shapley") + "".join("  G=%-5g" % g for g in gammas))
        for m in ALLM:
            print("%-18s" % m + "".join("  %6.3f" % v for v in methods[m][e]["mean"]))

    # --- PNGs ---
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Hajek": "#9467bd", "Direct": "#ff7f0e"}
        def sty(m):
            c = FAM[m.split("-")[0]]
            return (c, "-", "o") if m.endswith("-O-W") else (c, "-", "s") if m.endswith("-O-X") else (c, "--", None)
        # value vs Gamma, KNN vs Shapley
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
        for ax, e, ttl in [(axs[0], "knn", "KNN"), (axs[1], "shp", "Shapley")]:
            for m in ALLM:
                c, ls, mk = sty(m); ax.plot(gammas, methods[m][e]["mean"], ls, color=c, marker=mk, ms=5, lw=2, label=m)
            ax.axhline(out["oracle"]["mean"], color="#444", ls=":", label="oracle %.2f" % out["oracle"]["mean"])
            ax.axhline(out["never_treat"]["mean"], color="#bbb", ls=":", label="never-treat %.2f" % out["never_treat"]["mean"])
            ax.set_xlabel("Γ"); ax.set_title("Test realized outcome — %s extension" % ttl, fontsize=10); ax.grid(alpha=.25)
        axs[0].set_ylabel("test realised E[Y] (%d seeds)" % a.seeds); axs[1].legend(fontsize=6.5, ncol=2)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "%s_val.png" % stem), dpi=120); plt.close(fig)
        # extended policy curves at Gamma=2 (or nearest), small multiples
        gi2 = int(np.argmin([abs(g - 2.0) for g in gammas])); xg = np.linspace(-1, 1, a.ngrid)
        fig, axs = plt.subplots(2, 4, figsize=(15, 6.4)); axs = axs.ravel()
        for ai, m in enumerate(ALLM):
            ax = axs[ai]
            for e, ec in [("knn", "#1f77b4"), ("shp", "#d62728")]:
                gl = methods[m][e]["grid_mean"][gi2]
                if gl is not None: ax.plot(xg, gl, "-", color=ec, lw=2, label=e)
            ax.step(xg, (xg > 0).astype(float), where="post", color="#444", ls=":", lw=1.5, label="oracle")
            ax.set_title(m, fontsize=8.5); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=.25)
            if ai == 0: ax.legend(fontsize=7)
        for ai in range(len(ALLM), len(axs)): axs[ai].axis("off")
        fig.suptitle("Extended policy π(treat|X) vs X  (Γ=%g, mean over seeds)" % gammas[gi2], fontsize=11)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "%s_pol.png" % stem), dpi=120); plt.close(fig)
        print("saved %s_val.png, %s_pol.png" % (stem, stem), flush=True)
    except Exception as ex:
        print("PNG skipped: %s" % ex, flush=True)

if __name__ == "__main__":
    main()
