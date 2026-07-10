"""Continuous-X OWGAP runner — IN-SAMPLE realized-outcome comparison.

X is continuous (uniform on [-1,1], from assets/exp_owgap_cont/dgp.py). Each solver is called with
**dkw (raw per-unit support), and every learned policy is scored by its IN-SAMPLE realized
outcome on the training units' own known potential outcomes:

    V_hat(pi) = (1/n) * sum_i [ pi_1(X_i) * Y1_i + pi_0(X_i) * Y0_i ]        (Y1/Y0 = full["Y1"]/["Y0"])

and (lower-variance) the noise-free version with mu1/mu0 in place of Y1/Y0. Bypasses the grid-based
_tg / exact_value entirely. Runs 7 methods (all except IPW-O-W), uncapped only, over a Gamma sweep.

Usage:
  python3 assets/run_owgap_continuous.py --out assets/exp_owgap_cont/owgap_cont_insample.json \
      --n 400 --seeds 5 --gammas 1,1.5,2,3,4,6,8 --ceps 1.0 --workers 2 --threads 1
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DGP_PATH = str((ROOT / "assets/exp_owgap_cont/dgp.py").resolve())

XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["DoublyRobust-O-W"]                        # IPW-O-W intentionally EXCLUDED
ALLM = XX + OX + OW
_W = {}

def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)

def _init(dgp_path):
    import common; _W["common"] = common
    sp = importlib.util.spec_from_file_location("dgpmod", dgp_path); d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d
    S = {}
    suf, Cap = "uncapped", "Uncapped"          # uncapped regime only
    S["IPW-X-X"] = _load("methods/IPW-X-X/%s/ipw_x_x_%s.py" % (Cap, suf), "solve_ipw_x_x_%s" % suf)
    S["DoublyRobust-X-X"] = _load("methods/DoublyRobust-X-X/%s/doublyrobust_x_x_%s.py" % (Cap, suf), "solve_doublyrobust_x_x_%s" % suf)
    S["Direct-X-X"] = _load("methods/Direct-X-X/%s/direct_x_x_%s.py" % (Cap, suf), "solve_direct_x_x_%s" % suf)
    S["IPW-O-X"] = _load("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf)
    S["DoublyRobust-O-X"] = _load("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf)
    S["Hajek-O-X"] = _load("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (Cap, suf), "solve_hajek_o_x_%s" % suf)
    S["DoublyRobust-O-W"] = _load("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf)
    _W["S"] = S

def _insample(res, Y1, Y0):
    pi1 = np.asarray(res.pi[1], float).ravel()
    return float(np.mean(pi1 * Y1 + (1.0 - pi1) * Y0))

def solve_job(job):
    seed, N, gammas, ceps, mesh = job
    common = _W["common"]; d = _W["dgp"]; S = _W["S"]; K = d.K
    dkw = {"discretize": True, "mesh": int(mesh), "mesh_range": (-1.0, 1.0)} if mesh and int(mesh) > 0 else {"discretize": False}
    obs, full = d.generate(N, seed)
    X, T, Y = obs["X"], obs["T"], obs["Y"]
    Y1, Y0, mu1f, mu0f = full["Y1"], full["Y0"], full["mu1"], full["mu0"]
    w, _ = common.ipw_weights_from_data(X, T, K)
    wraw, _ = common.ipw_weights_from_data(X, T, K, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X)
    eps = tuple(common.tight_epsilon(Dm, T, w, K, is_distance=True, c_eps=ceps))
    ng = len(gammas)
    out = {m: {"val": [np.nan] * ng, "val_nf": [np.nan] * ng} for m in ALLM}

    def _store(m, gi, res):
        out[m]["val"][gi] = float(np.mean(res.pi[1] * Y1 + (1 - res.pi[1]) * Y0))
        out[m]["val_nf"][gi] = float(np.mean(res.pi[1] * mu1f + (1 - res.pi[1]) * mu0f))

    # --- Gamma-free (X-X): solve once, replicate across Gamma ---
    try:
        r = S["IPW-X-X"](X, T, Y, w, n_arms=K, **dkw)
        for gi in range(ng): _store("IPW-X-X", gi, r)
    except Exception as e: print("IPW-X-X seed%d FAIL: %s" % (seed, e), flush=True)
    try:
        r = S["DoublyRobust-X-X"](X, T, Y, w, mu, n_arms=K, **dkw)
        for gi in range(ng): _store("DoublyRobust-X-X", gi, r)
    except Exception as e: print("DoublyRobust-X-X seed%d FAIL: %s" % (seed, e), flush=True)
    try:
        r = S["Direct-X-X"](X, T, Y, n_arms=K, **dkw)
        for gi in range(ng): _store("Direct-X-X", gi, r)
    except Exception as e: print("Direct-X-X seed%d FAIL: %s" % (seed, e), flush=True)

    # --- Gamma-dependent (O-X, Hajek, DR-O-W) ---
    for gi, g in enumerate(gammas):
        for m, call in [
            ("IPW-O-X", lambda: S["IPW-O-X"](X, T, Y, w, n_arms=K, Gamma=g, **dkw)),
            ("DoublyRobust-O-X", lambda: S["DoublyRobust-O-X"](X, T, Y, w, mu, n_arms=K, Gamma=g, **dkw)),
            ("Hajek-O-X", lambda: S["Hajek-O-X"](X, T, Y, wraw, n_arms=K, Gamma=g, maximize=True, **dkw)),
            ("DoublyRobust-O-W", lambda: S["DoublyRobust-O-W"](X, T, Y, w, mu, n_arms=K, Gamma=g, **dkw, zscore=False, epsilon=eps)),
        ]:
            try:
                _store(m, gi, call())
            except Exception as e:
                print("%s seed%d g%s FAIL: %s" % (m, seed, g, e), flush=True)

    # --- references (Gamma-independent) ---
    orc = d.oracle_policy(np.asarray(X).ravel())
    refs = {
        "oracle": float(np.mean(orc * Y1 + (1 - orc) * Y0)),
        "oracle_nf": float(np.mean(orc * mu1f + (1 - orc) * mu0f)),
        "never_treat": float(np.mean(Y0)),
        "all_treat": float(np.mean(Y1)),
    }
    return seed, out, refs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/exp_owgap_cont/owgap_cont_insample.json")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gammas", default="1,1.5,2,3,4,6,8"); ap.add_argument("--ceps", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--mesh", type=int, default=0, help="0 = raw per-unit (discretize=False); >0 = snap X to this many levels on [-1,1]")
    a = ap.parse_args()
    gammas = [float(x) for x in a.gammas.split(",")]; seeds = list(range(a.seeds))
    Path("gurobi.env").write_text("Threads %d\n" % a.threads)
    jobs = [(sd, a.n, gammas, a.ceps, a.mesh) for sd in seeds]
    mode = ("mesh=%d levels" % a.mesh) if a.mesh > 0 else "raw per-unit (discretize=False)"
    print("continuous-X in-sample run: N=%d, %d seeds, gammas=%s, workers=%d (uncapped, %s)"
          % (a.n, a.seeds, gammas, a.workers, mode), flush=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(DGP_PATH,)) as pool:
        res = pool.map(solve_job, jobs)

    def agg(vals):
        arr = np.array(vals, float)
        return [round(float(np.nanmean(arr[:, gi])), 4) for gi in range(len(gammas))], \
               [round(float(np.nanstd(arr[:, gi])), 4) for gi in range(len(gammas))]

    methods = {}
    for m in ALLM:
        vmean, vsd = agg([o[m]["val"] for _, o, _ in res])
        vnf, _ = agg([o[m]["val_nf"] for _, o, _ in res])
        methods[m] = {"mean": vmean, "sd": vsd, "mean_nf": vnf}
    def refagg(k):
        vals = np.array([r[k] for _, _, r in res], float)
        return {"mean": round(float(np.mean(vals)), 4), "sd": round(float(np.std(vals)), 4)}
    out = {"N_train": a.n, "seeds": seeds, "gammas": gammas, "regime": "uncap",
           "discretize": bool(a.mesh > 0), "mesh": a.mesh, "x_dist": "uniform(-1,1)",
           "oracle": refagg("oracle"), "oracle_nf": refagg("oracle_nf"),
           "never_treat": refagg("never_treat"), "all_treat": refagg("all_treat"),
           "methods": methods}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("saved %s  (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)

    # --- comparison table ---
    hdr = "method            " + "".join("  G=%-5g" % g for g in gammas)
    print("\n=== in-sample realized outcome (mean over %d seeds) ===" % a.seeds)
    print("oracle=%.3f  never_treat=%.3f  all_treat=%.3f" % (out["oracle"]["mean"], out["never_treat"]["mean"], out["all_treat"]["mean"]))
    print(hdr)
    for m in ALLM:
        print("%-18s" % m + "".join("  %6.3f" % v for v in methods[m]["mean"]))

    # --- PNG ---
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Hajek": "#9467bd", "Direct": "#ff7f0e"}
        def style(m):
            c = FAM[m.split("-")[0]]
            if m.endswith("-O-W"): return c, "-", "o"
            if m.endswith("-O-X"): return c, "-", "s"
            return c, "--", None
        fig, ax = plt.subplots(figsize=(7.5, 4.6))
        for m in ALLM:
            c, ls, mk = style(m); ax.plot(gammas, methods[m]["mean"], ls, color=c, marker=mk, ms=5, lw=2, label=m)
        ax.axhline(out["oracle"]["mean"], color="#444", ls=":", label="oracle %.2f" % out["oracle"]["mean"])
        ax.axhline(out["never_treat"]["mean"], color="#bbb", ls=":", label="never-treat %.2f" % out["never_treat"]["mean"])
        ax.set_xlabel("Γ"); ax.set_ylabel("in-sample realised E[Y] (%d seeds)" % a.seeds)
        ax.set_title("Continuous-X OWGAP — in-sample realized outcome (N=%d, %s)" % (a.n, mode), fontsize=10)
        ax.grid(alpha=.25); ax.legend(fontsize=7, ncol=2)
        png = a.out[:-5] + ".png" if a.out.endswith(".json") else os.path.join(os.path.dirname(a.out), "val_cont_insample.png")
        fig.tight_layout(); fig.savefig(png, dpi=120); plt.close(fig)
        print("saved %s" % png, flush=True)
    except Exception as e:
        print("PNG skipped: %s" % e, flush=True)

if __name__ == "__main__":
    main()
