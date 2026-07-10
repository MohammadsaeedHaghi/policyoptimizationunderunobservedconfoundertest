"""Continuous-X OWGAP — LIPSCHITZ policy class (instead of unit-by-unit optimization).

The raw per-unit LP has one free π per training point → overfits noise → trash on test. Here we
constrain the policy to be L-Lipschitz in X:  |π(X_i) − π(X_j)| ≤ L·||X_i − X_j||.  For 1-D X this
reduces to consecutive-in-sorted-order constraints (O(n), exact), coupling nearby-X decisions so the
policy must vary smoothly instead of memorizing each point.

We add this to the simple direct-LP methods (IPW-X-X, DoublyRobust-X-X, Direct-X-X — exact objectives
reconstructed from the solvers), train on continuous X, deploy on a held-out test set via KNN, and
sweep L. L=∞ (no constraint) is the unit-by-unit baseline.

Usage:
  python3 assets/run_owgap_lipschitz.py --out assets/exp_owgap_cont/owgap_lipschitz.json \
      --n 400 --n-test 2000 --seeds 5 --k 50
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "extensions" / "KNN")); sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
import gurobipy as gp
from gurobipy import GRB
import common
from common.gurobi_env import configure_gurobi_license
from knn import extend_with_knn
from shapley import extend_with_shapley

def _load(rel, mod):
    s = importlib.util.spec_from_file_location(mod, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[mod] = m; s.loader.exec_module(m); return m

D = _load("assets/exp_owgap_cont/dgp.py", "dgpcont")
K = D.K
METHODS = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
LGRID = [None, 20.0, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25]   # None = ∞ (unit-by-unit)

def solve_lip(Xtr, T, Y, w, mu, method, L):
    """Exact IPW-X-X / DoublyRobust-X-X / Direct-X-X objective + optional L-Lipschitz constraint. Returns π_1 per unit."""
    n = len(T); configure_gurobi_license()
    m = gp.Model(); m.Params.OutputFlag = 0
    p = m.addVars(n, lb=0.0, ub=1.0, name="pi1")           # treat prob per unit (K=2: pi0 = 1 - pi1)
    obj = gp.LinExpr()
    for i in range(n):
        t = int(T[i]); pit = p[i] if t == 1 else (1 - p[i])   # π_{T_i}(X_i)
        if method == "IPW-X-X":
            obj += (1.0 / n) * float(Y[i] * w[i]) * pit
        elif method == "Direct-X-X":
            obj += (1.0 / n) * float(Y[i]) * pit               # unit weight ŵ≡1
        else:                                                  # DoublyRobust-X-X (AIPW)
            obj += (1.0 / n) * (float(mu[i, 1]) * p[i] + float(mu[i, 0]) * (1 - p[i]))
            obj += (1.0 / n) * float(w[i] * (Y[i] - mu[i, t])) * pit
    m.setObjective(obj, GRB.MAXIMIZE)
    if L is not None:                                          # 1-D Lipschitz: consecutive sorted pairs
        o = np.argsort(Xtr.ravel()); xs = Xtr.ravel()[o]
        for a in range(n - 1):
            i, j = int(o[a]), int(o[a + 1]); dx = float(xs[a + 1] - xs[a])
            m.addConstr(p[i] - p[j] <= L * dx); m.addConstr(p[j] - p[i] <= L * dx)
    m.optimize()
    return np.array([p[i].X for i in range(n)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="assets/exp_owgap_cont/owgap_lipschitz.json")
    ap.add_argument("--n", type=int, default=400); ap.add_argument("--n-test", type=int, default=2000, dest="ntest")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--k", type=int, default=50)
    ap.add_argument("--ngrid", type=int, default=121)
    a = ap.parse_args()
    seeds = list(range(a.seeds)); Path("gurobi.env").write_text("Threads 1\n")
    Xg = np.linspace(-1.0, 1.0, a.ngrid).reshape(-1, 1)
    Lkeys = ["inf" if L is None else ("%g" % L) for L in LGRID]
    t0 = time.time()
    per = {m: {lk: [] for lk in Lkeys} for m in METHODS}       # KNN test values per seed
    per_s = {m: {lk: [] for lk in Lkeys} for m in METHODS}     # Shapley test values per seed
    ins = {m: {lk: [] for lk in Lkeys} for m in METHODS}       # in-sample values per seed
    gridpol = {m: {} for m in METHODS}                         # seed0 grid policy per L (KNN, for plotting)
    gridpol_s = {m: {} for m in METHODS}                       # seed0 grid policy per L (Shapley)
    refs = {"oracle": [], "never_treat": [], "all_treat": []}
    for seed in seeds:
        obs, full = D.generate(a.n, seed); Xtr, T, Y = obs["X"], obs["T"], obs["Y"]
        te, ft = D.generate(a.ntest, seed + 1000); Xte = te["X"]; Y1t, Y0t = ft["Y1"], ft["Y0"]
        Y1tr, Y0tr = full["Y1"], full["Y0"]
        w, _ = common.ipw_weights_from_data(Xtr, T, K)
        mu = common.outcome_means(Xtr, T, Y, n_arms=K, cross_fit=True)
        orc = D.oracle_policy(Xte.ravel())
        refs["oracle"].append(float(np.mean(orc * Y1t + (1 - orc) * Y0t)))
        refs["never_treat"].append(float(np.mean(Y0t))); refs["all_treat"].append(float(np.mean(Y1t)))
        for meth in METHODS:
            for L, lk in zip(LGRID, Lkeys):
                pi1 = solve_lip(Xtr, T, Y, w, mu, meth, L)
                ins[meth][lk].append(float(np.mean(pi1 * Y1tr + (1 - pi1) * Y0tr)))
                pe = extend_with_knn(Xte, Xtr, pi1, k=a.k)
                per[meth][lk].append(float(np.mean(pe * Y1t + (1 - pe) * Y0t)))
                ps = extend_with_shapley(Xte, Xtr, pi1)
                per_s[meth][lk].append(float(np.mean(ps * Y1t + (1 - ps) * Y0t)))
                if seed == 0:
                    gridpol[meth][lk] = [round(float(v), 4) for v in extend_with_knn(Xg, Xtr, pi1, k=a.k)]
                    gridpol_s[meth][lk] = [round(float(v), 4) for v in extend_with_shapley(Xg, Xtr, pi1)]
        print("[%s] seed %d done (%.1f min)" % (time.strftime("%H:%M:%S"), seed, (time.time() - t0) / 60), flush=True)
    out = {"N_train": a.n, "N_test": a.ntest, "seeds": seeds, "k": a.k, "Lgrid": Lkeys,
           "grid": [round(float(x), 4) for x in Xg.ravel()],
           "oracle": round(float(np.mean(refs["oracle"])), 4),
           "never_treat": round(float(np.mean(refs["never_treat"])), 4),
           "all_treat": round(float(np.mean(refs["all_treat"])), 4),
           "methods": {m: {"test_mean": {lk: round(float(np.mean(per[m][lk])), 4) for lk in Lkeys},
                            "test_shp": {lk: round(float(np.mean(per_s[m][lk])), 4) for lk in Lkeys},
                            "test_sd": {lk: round(float(np.std(per[m][lk])), 4) for lk in Lkeys},
                            "insample_mean": {lk: round(float(np.mean(ins[m][lk])), 4) for lk in Lkeys},
                            "gridpol": gridpol[m], "gridpol_shp": gridpol_s[m]} for m in METHODS}}
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("saved %s  (%.1f min)" % (a.out, (time.time() - t0) / 60), flush=True)

    # --- table (KNN vs Shapley) ---
    print("\n=== TEST realized outcome vs L (mean over %d seeds) — oracle=%.3f never=%.3f ===" %
          (a.seeds, out["oracle"], out["never_treat"]))
    for ext, key in [("KNN", "test_mean"), ("Shapley", "test_shp")]:
        print("-- %s --  " % ext + "".join("  %7s" % lk for lk in Lkeys))
        for m in METHODS:
            print("%-18s" % m + "".join("  %7.3f" % out["methods"][m][key][lk] for lk in Lkeys))

    # --- PNGs ---
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        outdir = os.path.dirname(a.out)
        FAM = {"IPW-X-X": "#1f77b4", "DoublyRobust-X-X": "#2ca02c", "Direct-X-X": "#ff7f0e"}
        Lnum = [np.nan if lk == "inf" else float(lk) for lk in Lkeys]
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for m in METHODS:
            ys = [out["methods"][m]["test_mean"][lk] for lk in Lkeys]
            ax.plot(range(len(Lkeys)), ys, "-o", color=FAM[m], lw=2, ms=5, label=m)
        ax.axhline(out["oracle"], color="#444", ls=":", label="oracle %.2f" % out["oracle"])
        ax.axhline(out["never_treat"], color="#bbb", ls=":", label="never-treat %.2f" % out["never_treat"])
        ax.set_xticks(range(len(Lkeys))); ax.set_xticklabels(Lkeys); ax.set_xlabel("Lipschitz L  (∞ = unit-by-unit)")
        ax.set_ylabel("test realised E[Y] (%d seeds)" % a.seeds); ax.grid(alpha=.25); ax.legend(fontsize=8)
        ax.set_title("Lipschitz policy class fixes the continuous-X overfitting", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "lipschitz_val.png"), dpi=120); plt.close(fig)
        # policy shapes (seed0) for a few L on DoublyRobust-X-X
        xg = Xg.ravel(); fig, ax = plt.subplots(figsize=(7.2, 4.4))
        for lk, col in [("inf", "#d62728"), ("5", "#2ca02c"), ("1", "#1f77b4"), ("0.5", "#9467bd")]:
            gp2 = out["methods"]["DoublyRobust-X-X"]["gridpol"].get(lk)
            if gp2: ax.plot(xg, gp2, "-", lw=2, label="L=%s" % lk)
        ax.step(xg, (xg > 0).astype(float), where="post", color="#444", ls=":", lw=1.5, label="oracle")
        ax.set_xlabel("X"); ax.set_ylabel("π(treat | X)"); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=.25); ax.legend(fontsize=8)
        ax.set_title("DoublyRobust-X-X learned policy vs X, by Lipschitz L (seed 0)", fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(outdir, "lipschitz_pol.png"), dpi=120); plt.close(fig)
        print("saved lipschitz_val.png, lipschitz_pol.png", flush=True)
    except Exception as e:
        print("PNG skipped: %s" % e, flush=True)

if __name__ == "__main__":
    main()
