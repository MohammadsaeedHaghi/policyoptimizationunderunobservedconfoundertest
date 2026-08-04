#!/usr/bin/env python3
"""Run every method on one RCT-derived benchmark: one (dataset, seed, c_eps) cell per job.

One SLURM array task = one dataset. Within a task, the (seed, c_eps) cells are farmed out to a
process pool.

Grid (as requested): Gamma in {1.5, 2, 4, 8} (4 values; 4 == the DGP's exact Lambda),
L in {inf, 3, 1}, c_eps in {1, 2}, NO capacity constraint.

Methods
  robust, Gamma-swept : IPW-O-X, DoublyRobust-O-X, Hajek-O-X          (c_eps-free)
                        IPW-O-W, DoublyRobust-O-W, Hajek-O-W          (c_eps-swept)
  baselines, Gamma-swept : SharpIPW-O-X (Hess et al. 2502.13022), Kallus (regret form)
  baselines, Gamma-free  : IPW-X-X, DoublyRobust-X-X, Direct-X-X, naive-DR, never/all-treat, oracle

Per seed the treatment vector, both potential outcomes and the 300-row training sample are all
REDRAWN from the saved population arrays, so the seed spread is genuine sampling variation.
The learner sees only (x, T, Y_T) plus an ESTIMATED propensity; u and S are never passed to it.

Every raw pointwise policy is persisted (save-everything rule).

Usage: python3 assets/uci10/run_uci.py --data NAME --out OUT.json [--seeds 5] [--workers 8]
"""
import sys, os, json, argparse, time, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path("/home1/haghim/code 1.1")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
sys.path.insert(0, str(ROOT / "methods" / "SharpHess"))

GAMMAS = [1.0, 2.0, 4.0, 8.0]
LGRID = [None, 3.0, 1.0]
CEPS = [1.0, 2.0]
OX = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X"]
OW = ["IPW-O-W", "DoublyRobust-O-W", "Hajek-O-W"]
XX = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X"]
N_TRAIN, N_TEST = 300, 4000
PGRID = np.linspace(-1.0, 1.0, 41)
_W = {}


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def _init(npz_path):
    import common
    from shapley import extract_support
    _W["common"] = common; _W["extract"] = extract_support
    S = {}
    S["IPW-O-X"] = _load("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
    S["DoublyRobust-O-X"] = _load("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped")
    S["Hajek-O-X"] = _load("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py", "solve_hajek_o_x_uncapped")
    S["IPW-O-W"] = _load("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
    S["DoublyRobust-O-W"] = _load("methods/DoublyRobust-O-W/Uncapped/doublyrobust_o_w_uncapped.py", "solve_doublyrobust_o_w_uncapped")
    S["Hajek-O-W"] = _load("methods/Hajek-O-W/Uncapped/hajek_o_w_uncapped.py", "solve_hajek_o_w_uncapped")
    S["IPW-X-X"] = _load("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped")
    S["DoublyRobust-X-X"] = _load("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped")
    S["Direct-X-X"] = _load("methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py", "solve_direct_x_x_uncapped")
    _W["S"] = S
    _W["kallus"] = _load("methods/Kallus/kallus.py", "fit_kallus")
    _W["kpred"] = _load("methods/Kallus/kallus.py", "predict_kallus")
    import sharp_hess
    _W["hess"] = sharp_hess
    z = np.load(npz_path)
    _W["pop"] = {k: z[k] for k in z.files}


def _shapley_fast(Xnew, sX, sp, block=200):
    Xnew = np.asarray(Xnew, float).reshape(-1, 1)
    sX = np.asarray(sX, float).reshape(-1, 1); sp = np.asarray(sp, float).ravel()
    out = np.empty(len(Xnew)); dg = np.arange(len(sX))
    for s0 in range(0, len(Xnew), block):
        xb = Xnew[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        Sm = dd[:, :, None] + dd[:, None, :]; Sm = np.where(Sm > 0, Sm, 1.0)
        A = (dd[:, None, :] * sp[None, :, None] + dd[:, :, None] * sp[None, None, :]) / Sm
        A[:, dg, dg] = sp[None, :]
        v = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0
        if ex.any(): v[ex] = sp[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = v
    return out


def draw(seed):
    """Redraw only the ASSIGNMENT and the train/test split. Y0/Y1 are the dataset's own."""
    P = _W["pop"]; rng = np.random.default_rng(10_000 + seed)
    x, e, Y0, Y1 = P["x"], P["e"], P["Y0"], P["Y1"]
    n = len(x)
    T = (rng.uniform(size=n) < e).astype(int)
    Yo = np.where(T == 1, Y1, Y0)
    perm = rng.permutation(n)
    tr, te = perm[:N_TRAIN], perm[N_TRAIN:N_TRAIN + N_TEST]
    return dict(Xtr=x[tr].reshape(-1, 1), Ttr=T[tr], Ytr=Yo[tr],
                Xte=x[te], Y0te=Y0[te], Y1te=Y1[te], cate_te=(Y1 - Y0)[te])


def cell(job):
    """One (seed, c_eps) cell: the whole Gamma x L grid for every method."""
    seed, ceps = job
    common = _W["common"]; S = _W["S"]; K = 2
    d = draw(seed)
    X, T, Yraw = d["Xtr"], d["Ttr"], d["Ytr"]
    Xte, Y1t, Y0t = d["Xte"], d["Y1te"], d["Y0te"]

    # CENTRE-AND-SCALE THE TRAINING OUTCOME. Not cosmetic: V(pi) = E[pi Y1 + (1-pi) Y0] shifts by
    # a constant when both potential outcomes shift, so the OPTIMAL POLICY is unchanged -- but the
    # worst-case program is not shift-invariant. With a non-negative outcome (these are 0/1 for the
    # binary datasets) and a free pi per support point, every unit's contribution has a fixed sign,
    # so the LP degenerates to pi_i = 1{Y_i > 0} and becomes EXACTLY Gamma-invariant: the box stops
    # binding altogether. Verified on the aids_clinical pilot -- all four Gammas returned the same
    # 2-valued policy, identical to the unconfoundedness-assuming X-X solution. Hess et al.
    # standardise internally for the same reason, so this also puts every arm on one footing.
    # Evaluation below always uses the RAW test outcomes, so reported values stay on the data scale.
    _ym, _ys = float(np.mean(Yraw)), float(np.std(Yraw)) or 1.0
    Y = (Yraw - _ym) / _ys

    def dep(Xnew, vals):
        sX, sp = _W["extract"](X, vals)
        return _shapley_fast(Xnew, sX, sp)

    def val(vals):
        pe = dep(Xte, vals); return float(np.mean(pe * Y1t + (1 - pe) * Y0t))

    w, _ = common.ipw_weights_from_data(X, T, K)                      # ESTIMATED propensity
    wraw, _ = common.ipw_weights_from_data(X, T, K, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=K, cross_fit=True)
    Dm = common.pairwise_distance_matrix(X)
    eps = tuple(common.tight_epsilon(Dm, T, w, K, is_distance=True, c_eps=ceps))

    Gk = ["%g" % g for g in GAMMAS]
    Lk = ["inf" if L is None else "%g" % L for L in LGRID]
    meths = (OX if ceps == CEPS[0] else []) + OW          # O-X is c_eps-free: solve it once
    grid = {m: {g: {l: float("nan") for l in Lk} for g in Gk} for m in meths}
    pol = {m: {g: {} for g in Gk} for m in meths}
    sup = {m: {g: {} for g in Gk} for m in meths}

    def call(m, g, L):
        if m == "IPW-O-X": return S[m](X, T, Y, w, n_arms=K, Gamma=g, discretize=False, lipschitz=L)
        if m == "DoublyRobust-O-X": return S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, lipschitz=L)
        if m == "Hajek-O-X": return S[m](X, T, Y, wraw, n_arms=K, Gamma=g, maximize=True, discretize=False, lipschitz=L)
        if m == "IPW-O-W": return S[m](X, T, Y, w, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=L)
        if m == "DoublyRobust-O-W": return S[m](X, T, Y, w, mu, n_arms=K, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=L)
        return S[m](X, T, Y, w, n_arms=K, Gamma=g, maximize=True, discretize=False, zscore=False, epsilon=eps, lipschitz=L)

    for m in meths:
        for g in GAMMAS:
            for L, lk in zip(LGRID, Lk):
                try:
                    r = call(m, g, L)
                    grid[m]["%g" % g][lk] = val(r.pi[1])
                    pol[m]["%g" % g][lk] = [round(float(v), 4) for v in dep(PGRID, r.pi[1])]
                    sup[m]["%g" % g][lk] = [round(float(v), 4) for v in r.pi[1]]
                except Exception as ex:
                    print("FAIL %s G=%s L=%s ce=%g seed=%d: %s" % (m, g, lk, ceps, seed, str(ex)[:90]), flush=True)

    out = {"grid": grid, "policies": pol, "support_pi": sup, "epsilon": [float(e) for e in eps]}

    # ---- Gamma-free and baseline arms: only on the first c_eps pass -------------------------
    if ceps == CEPS[0]:
        xx = {}
        for m in XX:
            xx[m] = {}
            for L, lk in zip(LGRID, Lk):
                try:
                    if m == "IPW-X-X": r = S[m](X, T, Y, w, n_arms=K, discretize=False, lipschitz=L)
                    elif m == "DoublyRobust-X-X": r = S[m](X, T, Y, w, mu, n_arms=K, discretize=False, lipschitz=L)
                    else: r = S[m](X, T, Y, n_arms=K, discretize=False, lipschitz=L)
                    xx[m][lk] = {"value": val(r.pi[1]),
                                 "curve": [round(float(v), 4) for v in dep(PGRID, r.pi[1])]}
                except Exception as ex:
                    print("FAIL %s L=%s seed=%d: %s" % (m, lk, seed, str(ex)[:90]), flush=True)
        out["xx"] = xx

        hess = {}
        for g in GAMMAS:
            try:
                sc = _W["hess"].fit_scores(X, T, Y, Gamma=g, k=15, n_folds=2, maximize=True, seed=seed)
                th = _W["hess"].learn_policy_parametric(X, sc, n_iter=300, lr=0.05, restarts=3,
                                                        seed=seed, maximize=True)
                pe = _W["hess"].apply_policy(th, Xte.reshape(-1, 1))
                hess["%g" % g] = {"value": float(np.mean(pe * Y1t + (1 - pe) * Y0t)),
                                  "curve": [round(float(v), 4) for v in
                                            _W["hess"].apply_policy(th, PGRID.reshape(-1, 1))]}
            except Exception as ex:
                print("FAIL Hess G=%g seed=%d: %s" % (g, seed, str(ex)[:90]), flush=True)
        out["hess"] = hess

        kal = {}
        for g in GAMMAS:
            try:
                r = _W["kallus"](X, T, Y, wraw, n_arms=K, Gamma=g, maximize=True, seed=seed)
                pe = _W["kpred"](r.theta, Xte.reshape(-1, 1))[:, 1]
                kal["%g" % g] = {"value": float(np.mean(pe * Y1t + (1 - pe) * Y0t)),
                                 "curve": [round(float(v), 4) for v in
                                           _W["kpred"](r.theta, PGRID.reshape(-1, 1))[:, 1]]}
            except Exception as ex:
                print("FAIL Kallus G=%g seed=%d: %s" % (g, seed, str(ex)[:90]), flush=True)
        out["kallus"] = kal

        nv = (mu[:, 1] - mu[:, 0] > 0).astype(float)
        orc = (d["cate_te"] > 0).astype(float)
        out["refs"] = {"oracle": float(np.mean(orc * Y1t + (1 - orc) * Y0t)),
                       "never_treat": float(np.mean(Y0t)), "all_treat": float(np.mean(Y1t)),
                       "naive_dr": val(nv), "P_T1": float(T.mean())}
        out["naive_curve"] = [round(float(v), 4) for v in dep(PGRID, nv)]
        out["X_train"] = [round(float(v), 5) for v in X.ravel()]
        out["T_train"] = [int(v) for v in T]
        out["Y_train_raw"] = [round(float(v), 5) for v in Yraw]
        out["Y_standardisation"] = {"mean": _ym, "sd": _ys}
    return (seed, ceps, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ceps", type=float, required=True)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    npz = str(here / "prepared" / (a.data + "_pop.npz"))
    meta = [m for m in json.loads((here / "prepared" / "_index.json").read_text())
            if m["name"] == a.data][0]
    t0 = time.time()
    _init(npz)
    seed, ceps, o = cell((a.seed, a.ceps))
    res = {"s%d_ce%g" % (seed, ceps): o}
    print("  done seed=%d ceps=%g  (%.0fs)" % (seed, ceps, time.time() - t0), flush=True)

    payload = {"dataset": a.data, "meta": meta, "gammas": GAMMAS,
               "lipschitz": ["inf" if L is None else "%g" % L for L in LGRID],
               "ceps": [a.ceps], "n_train": N_TRAIN, "n_test": N_TEST, "seed": a.seed,
               "policy_grid": [round(float(v), 4) for v in PGRID], "cells": res,
               "elapsed_s": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(a.out, "w"))
    print("saved %s (%.0fs)" % (a.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
