#!/usr/bin/env python3
"""N=500 full grid as a SLURM JOB ARRAY: one task = one (campaign, seed, c_eps, cap) slice.

WHY THIS GRANULARITY (measured, not guessed). The request was one job per
(method, Gamma, L, c_eps). Two cluster facts make that strictly slower:

  * MaxJobsPU = 100 -- only 100 jobs RUN at once, however many are submitted (limit 5000).
    So one-solve-per-job caps concurrency at 100 solves. Packing tasks with W workers each
    gives 100*W concurrent solves; at W=8 that is 800.
  * Per-job overhead is not negligible next to the solve. Python start + imports + data gen +
    the tight_epsilon LP is ~4 s, against a mean solve of 6.4 s. One-solve-per-job would spend
    ~40% of the campaign on setup and re-run the SAME epsilon LP tens of thousands of times.

Measured at n=500 (debug node): IPW-O-X 0.06 s, DR-O-X 0.04 s, Hajek-O-X 0.07 s,
IPW-O-W 16.7 s, DR-O-W 15.3 s -> 32.2 s per (Gamma, L, c_eps, cap) cell for all five methods.
The O-W pair is 99.6% of the cost, so the useful thing to parallelise is Gamma x L, which is
what this does -- every (method, Gamma, L, c_eps, cap) combination is still computed and stored
separately, exactly as asked; only the JOB packing differs.

Full grid: 3 campaigns x 10 seeds x ~9.5 Gamma x 8 L x 3 c_eps x 4 cap x 5 methods
         = 136,800 solves ~ 245 CPU-hours.
Array of 360 tasks x 8 workers, throttled to 100 concurrent -> ~20-40 min wall.

Usage (array): python3 assets/grand/run_task500.py --task-id $SLURM_ARRAY_TASK_ID
"""
import sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))

CAMPS = [("gs", "assets/exp_gstar/dgp_cont.py", [1, 2, 3, 5, 8, 12, 16, 24, 32]),
         ("km", "assets/exp_msmbench/dgp_g15.py", [1, 1.6487, 2.7183, 4.4817, 7.3891, 12, 16, 24, 32]),
         ("kz", "assets/exp_kz18/dgp.py", [1, 1.6487, 2.7183, 4.4817, 7.3891, 12, 16, 24, 32, 40])]
SEEDS = list(range(10))
CEPS = [0.5, 1.0, 2.0]
CAPS = [None, 0.3, 0.4, 0.5]
LGRID = [None, 10.0, 5.0, 3.0, 2.0, 1.5, 1.0, 0.5]
METHODS = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
PGRID = np.linspace(-1.0, 1.0, 41)
_W = {}


def tasks():
    """Flat task list -> (campaign, dgp, gammas, seed, c_eps, cap). 3*10*3*4 = 360."""
    out = []
    for camp, dgp, gg in CAMPS:
        for sd in SEEDS:
            for ce in CEPS:
                for cp in CAPS:
                    out.append((camp, dgp, gg, sd, ce, cp))
    return out


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def _init(dgp_path, capped):
    import common
    _W["common"] = common
    from shapley import extract_support
    _W["extract"] = extract_support
    sp = importlib.util.spec_from_file_location("dgpmod", dgp_path)
    d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d
    suf, Cap = ("capped", "Capped") if capped else ("uncapped", "Uncapped")
    _W["S"] = {
        "IPW-O-X": _load("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf),
        "DoublyRobust-O-X": _load("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf),
        "Hajek-O-X": _load("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (Cap, suf), "solve_hajek_o_x_%s" % suf),
        "IPW-O-W": _load("methods/IPW-O-W/%s/ipw_o_w_%s.py" % (Cap, suf), "solve_ipw_o_w_%s" % suf),
        "DoublyRobust-O-W": _load("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf)}


def _shap(Xn, sX, spv, block=200):
    Xn = np.asarray(Xn, float).reshape(-1, 1)
    out = np.empty(len(Xn)); dg = np.arange(len(sX))
    for s0 in range(0, len(Xn), block):
        xb = Xn[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        S = dd[:, :, None] + dd[:, None, :]; S = np.where(S > 0, S, 1.0)
        A = (dd[:, None, :] * spv[None, :, None] + dd[:, :, None] * spv[None, None, :]) / S
        A[:, dg, dg] = spv[None, :]
        v = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0
        if ex.any(): v[ex] = spv[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = v
    return out


def one_cell(arg):
    """One (Gamma, L) cell: all 5 methods. The unit of parallelism inside a task.

    Parallelising over Gamma alone gave only 9-10 units per task, which capped useful workers
    at ~8 and left us using 796 of the 2000 cores the account allows. (Gamma, L) gives ~76 units,
    so a task can absorb 16+ workers and the array can actually saturate the CPU allowance.
    """
    g, Lv, payload = arg
    (X, T, Y, Xte, Y1, Y0, w, wraw, mu, eps, cap) = payload
    common = _W["common"]; S = _W["S"]; xsup = _W["extract"]
    kw = {"cap": (1.0, float(cap))} if cap is not None else {}
    lk = "inf" if Lv is None else "%g" % Lv
    res = {}
    for m in METHODS:
        res[m] = {}
        if True:
            try:
                if m == "IPW-O-X": r = S[m](X, T, Y, w, n_arms=2, Gamma=g, discretize=False, lipschitz=Lv, **kw)
                elif m == "DoublyRobust-O-X": r = S[m](X, T, Y, w, mu, n_arms=2, Gamma=g, discretize=False, lipschitz=Lv, **kw)
                elif m == "Hajek-O-X": r = S[m](X, T, Y, wraw, n_arms=2, Gamma=g, maximize=True, discretize=False, lipschitz=Lv, **kw)
                elif m == "IPW-O-W": r = S[m](X, T, Y, w, n_arms=2, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=Lv, **kw)
                else: r = S[m](X, T, Y, w, mu, n_arms=2, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=Lv, **kw)
                sX, spv = xsup(X, r.pi[1])
                pe = _shap(Xte, np.asarray(sX, float), np.asarray(spv, float).ravel())
                res[m][lk] = {"value": float(np.mean(pe * Y1 + (1 - pe) * Y0)),
                              "obj": float(r.objective_value),
                              "pi_support": [round(float(v), 4) for v in r.pi[1]]}
            except Exception as ex:
                print("FAIL %s G=%s L=%s: %s" % (m, g, lk, ex), flush=True)
                res[m][lk] = {"value": float("nan")}
    return "%g" % g, lk, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--n-test", type=int, default=4000, dest="nte")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--outdir", default="assets/grand500")
    a = ap.parse_args()
    TL = tasks()
    if not (0 <= a.task_id < len(TL)):
        raise SystemExit("task-id must be in [0, %d)" % len(TL))
    camp, dgp, gg, seed, ceps, cap = TL[a.task_id]
    tag = "%s_s%d_ce%g_%s" % (camp, seed, ceps, "uncap" if cap is None else "cap%g" % (100 * cap))
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / ("%s.json" % tag)
    if out_path.exists():
        print("already done:", out_path); return
    Path("gurobi.env").write_text("Threads 1\n")
    print("task %d -> %s  n=%d gammas=%s L=%d methods=%d workers=%d"
          % (a.task_id, tag, a.n, gg, len(LGRID), len(METHODS), a.workers), flush=True)

    import common
    sp = importlib.util.spec_from_file_location("dgp_main", str(ROOT / dgp))
    d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d)
    obs, _f = d.generate(a.n, seed); X, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(a.nte, seed + 1000); Xte = te["X"]; Y1, Y0 = ft["Y1"], ft["Y0"]
    w, _ = common.ipw_weights_from_data(X, T, 2)
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
    t0 = time.time()
    Dm = common.pairwise_distance_matrix(X)
    eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=ceps))
    print("  eps=%s (%.1fs)" % (tuple(round(e, 5) for e in eps), time.time() - t0), flush=True)
    payload = (X, T, Y, Xte, Y1, Y0, w, wraw, mu, eps, cap)

    with mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                      initargs=(str((ROOT / dgp).resolve()), cap is not None)) as pool:
        res = pool.map(one_cell, [(float(g), Lv, payload) for g in gg for Lv in LGRID])
    merged = {}
    for gk, lk, r in res:
        for m in METHODS:
            merged.setdefault(gk, {}).setdefault(m, {})[lk] = r[m][lk]
    orc = d.oracle_policy(Xte.ravel())
    nv = (mu[:, 1] - mu[:, 0] > 0).astype(float)
    out = {"campaign": camp, "seed": seed, "ceps": ceps,
           "cap": ("uncap" if cap is None else float(cap)), "N_train": a.n, "N_test": a.nte,
           "gammas": ["%g" % g for g in gg], "Lgrid": ["inf" if L is None else "%g" % L for L in LGRID],
           "methods": METHODS, "epsilon": [float(e) for e in eps],
           "refs": {"oracle": float(np.mean(orc * Y1 + (1 - orc) * Y0)),
                    "never_treat": float(np.mean(Y0)), "all_treat": float(np.mean(Y1))},
           "naive_support": [round(float(v), 4) for v in nv],
           "X_support": [round(float(v), 5) for v in np.asarray(X).ravel()],
           "cells": merged}
    out_path.write_text(json.dumps(out))
    print("saved %s (%.1f min)" % (out_path, (time.time() - t0) / 60), flush=True)
    print("DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
