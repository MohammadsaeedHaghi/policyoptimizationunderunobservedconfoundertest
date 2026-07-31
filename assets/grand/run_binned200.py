#!/usr/bin/env python3
"""n=200: give O-W (and O-X) SHARP'S OWN POLICY CLASS -- 15 quantile bins -- and re-measure.

Every campaign so far ran our solvers with a free pi per support point (up to n parameters),
against a sharp baseline that is piecewise-constant over 15 bins (~15 parameters). That is a
policy-class mismatch, not an uncertainty-set comparison, and it is the leading explanation for
why sharp wins nearly everywhere while every knob that helps us (Gamma=32, L=1, c_eps=0.5) is a
regulariser.

This isolates it. The binned arms snap X to the SAME 15 quantile-bin centres sharp uses, so
`common.support.tie_groups` (called unconditionally by every solver on support_X) pools pi
within each cell. The uncertainty set -- per-unit MSM box, n free adversarial weights, the
Wasserstein ball -- is untouched. Only the policy is coarsened.

Quantile snapping is done here rather than via the solvers' `discretize=True`, whose
`snap_to_grid` lays down an EQUALLY SPACED grid: that would not match sharp's quantile bins on a
non-uniform covariate such as the KZ18 index.

One task = one (campaign, seed). Reference: Sharp-O-X in assets/grand/sharp_all.json (n=200,
10 seeds, leak-free train-derived bins); naives in assets/grand/capped_refs.json.
"""
import sys, json, time, argparse, importlib.util
from pathlib import Path
import numpy as np
import multiprocessing as mp

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))

BINS = 15
import os
_ONLY = os.environ.get("BINNED_ONLY", "")          # e.g. "km" to run a single campaign
CAMPS = [("gs", "assets/exp_gstar/dgp_cont.py", [1, 2, 3, 5, 8, 12, 16, 24, 32]),
         ("km", "assets/exp_msmbench/dgp_g15.py", [1, 1.6487, 2.7183, 4.4817, 7.3891, 12, 16, 24, 32]),
         ("kz", "assets/exp_kz18/dgp.py", [1, 1.6487, 2.7183, 4.4817, 7.3891, 12, 16, 24, 32])]
if _ONLY:
    CAMPS = [c for c in CAMPS if c[0] == _ONLY]
METHODS = ["IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
#          arm name        binned?  lipschitz
ARMS = [("free-pi_L3", False, 3.0),        # what every campaign used (control)
        ("free-pi_L1", False, 1.0),        # control, heavier smoothing
        ("binned15_Lnone", True, None),    # sharp's class; binning is the only regulariser
        ("binned15_L3", True, 3.0)]        # sharp's class + smoothing
CEPS = [0.5, 1.0]
CAPS = [None, 0.3]
_W = {}


def _load(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel))
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)


def _init(dgp_path, gammas):
    import common
    _W["common"] = common
    _W["gammas"] = list(gammas)
    from shapley import extract_support
    _W["extract"] = extract_support
    sp = importlib.util.spec_from_file_location("dgpmod", dgp_path)
    d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d); _W["dgp"] = d
    for tag, (suf, Cap) in (("uncap", ("uncapped", "Uncapped")), ("cap", ("capped", "Capped"))):
        _W[tag] = {
            "IPW-O-X": _load("methods/IPW-O-X/%s/ipw_o_x_%s.py" % (Cap, suf), "solve_ipw_o_x_%s" % suf),
            "DoublyRobust-O-X": _load("methods/DoublyRobust-O-X/%s/doublyrobust_o_x_%s.py" % (Cap, suf), "solve_doublyrobust_o_x_%s" % suf),
            "Hajek-O-X": _load("methods/Hajek-O-X/%s/hajek_o_x_%s.py" % (Cap, suf), "solve_hajek_o_x_%s" % suf),
            "IPW-O-W": _load("methods/IPW-O-W/%s/ipw_o_w_%s.py" % (Cap, suf), "solve_ipw_o_w_%s" % suf),
            "DoublyRobust-O-W": _load("methods/DoublyRobust-O-W/%s/doublyrobust_o_w_%s.py" % (Cap, suf), "solve_doublyrobust_o_w_%s" % suf)}


def _shapley_fast(Xnew, sX, sp, block=200):
    """Same operator every previous campaign deployed with (kept local to avoid an import that
    would drag in capped_refs' module-level work)."""
    Xnew = np.asarray(Xnew, float).reshape(-1, 1)
    out = np.empty(len(Xnew)); dg = np.arange(len(sX))
    for s0 in range(0, len(Xnew), block):
        xb = Xnew[s0:s0 + block]
        dd = np.sqrt(((xb[:, None, :] - sX[None, :, :]) ** 2).sum(-1))
        S = dd[:, :, None] + dd[:, None, :]; S = np.where(S > 0, S, 1.0)
        A = (dd[:, None, :] * sp[None, :, None] + dd[:, :, None] * sp[None, None, :]) / S
        A[:, dg, dg] = sp[None, :]
        v = A.max(axis=1).min(axis=1)
        ex = dd.min(axis=1) <= 0
        if ex.any(): v[ex] = sp[dd[ex].argmin(axis=1)]
        out[s0:s0 + block] = v
    return out


def qbins(x, bins=BINS):
    """Sharp's binning: quantile edges from TRAIN x; each cell represented by its mean x."""
    ed = np.quantile(x, np.linspace(0, 1, bins + 1)); ed[0] -= 1e-9; ed[-1] += 1e-9
    ci = np.clip(np.digitize(x, ed) - 1, 0, bins - 1)
    ctr = np.array([x[ci == j].mean() if (ci == j).any() else 0.5 * (ed[j] + ed[j + 1])
                    for j in range(bins)])
    return ed, ci, ctr


def run_cell(job):
    seed, n, nte, ceps, cap = job
    common = _W["common"]; d = _W["dgp"]; xsup = _W["extract"]; GAMMAS = _W["gammas"]
    S = _W["cap"] if cap is not None else _W["uncap"]
    kw0 = {"cap": (1.0, float(cap))} if cap is not None else {}

    obs, _f = d.generate(n, seed); X, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(nte, seed + 1000); Xte = te["X"]; Y1, Y0 = ft["Y1"], ft["Y0"]
    x = np.asarray(X, float).ravel(); xte = np.asarray(Xte, float).ravel()
    ed, ci, ctr = qbins(x)
    cte = np.clip(np.digitize(xte, ed) - 1, 0, BINS - 1)
    Xb = ctr[ci].reshape(-1, 1)                       # snapped -> tie_groups pools pi per cell
    w, _ = common.ipw_weights_from_data(X, T, 2)
    wraw, _ = common.ipw_weights_from_data(X, T, 2, normalize=False)
    mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)

    out, diag = {}, {}
    for arm, binned, Lv in ARMS:
        XX = Xb if binned else X
        diag[arm] = int(len({round(float(v), 6) for v in np.asarray(XX).ravel()}))
        Dm = common.pairwise_distance_matrix(XX)
        try:
            eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=ceps))
        except Exception as ex:
            print("eps FAIL %s ceps=%s: %s" % (arm, ceps, ex), flush=True); continue
        out[arm] = {}
        for m in METHODS:
            out[arm][m] = {}
            for g in GAMMAS:
                # Gamma=1 with c_eps<1 is empty BY CONSTRUCTION (ball tighter than the nominal
                # radius while the box is a single point) -> skip rather than log a failure.
                if ceps < 1.0 and abs(g - 1.0) < 1e-9 and m.endswith("O-W"):
                    out[arm][m]["%g" % g] = float("nan"); continue
                try:
                    if m == "IPW-O-X": r = S[m](XX, T, Y, w, n_arms=2, Gamma=g, discretize=False, lipschitz=Lv, **kw0)
                    elif m == "DoublyRobust-O-X": r = S[m](XX, T, Y, w, mu, n_arms=2, Gamma=g, discretize=False, lipschitz=Lv, **kw0)
                    elif m == "Hajek-O-X": r = S[m](XX, T, Y, wraw, n_arms=2, Gamma=g, maximize=True, discretize=False, lipschitz=Lv, **kw0)
                    elif m == "IPW-O-W": r = S[m](XX, T, Y, w, n_arms=2, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=Lv, **kw0)
                    else: r = S[m](XX, T, Y, w, mu, n_arms=2, Gamma=g, discretize=False, zscore=False, epsilon=eps, lipschitz=Lv, **kw0)
                    pi = np.asarray(r.pi[1], float)
                    if binned:                        # cell policy -> deploy by cell membership
                        cellpi = np.array([pi[ci == j].mean() if (ci == j).any() else 0.0
                                           for j in range(BINS)])
                        pe = cellpi[cte]
                    else:                             # Shapley, as in every earlier campaign
                        sX, spv = xsup(X, pi)
                        pe = _shapley_fast(xte, np.asarray(sX, float), np.asarray(spv, float).ravel())
                    out[arm][m]["%g" % g] = float(np.mean(pe * Y1 + (1 - pe) * Y0))
                except Exception as ex:
                    print("FAIL %s %s G=%s ceps=%s cap=%s: %s" % (arm, m, g, ceps, cap, ex), flush=True)
                    out[arm][m]["%g" % g] = float("nan")
    orc = d.oracle_policy(xte)
    key = "ce%g_%s" % (ceps, "uncap" if cap is None else "cap30")
    print("  %s done | distinct support per arm: %s" % (key, diag), flush=True)
    return key, out, {"oracle": float(np.mean(orc * Y1 + (1 - orc) * Y0)),
                      "never": float(np.mean(Y0)), "cells": diag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=4000, dest="nte")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--outdir", default="assets/binned200")
    a = ap.parse_args()
    TL = [(c, p, g, sd) for c, p, g in CAMPS for sd in range(10)]
    if not (0 <= a.task_id < len(TL)):
        raise SystemExit("task-id must be in [0, %d)" % len(TL))
    camp, dgp, gg, seed = TL[a.task_id]
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    op = outdir / ("%s_s%d.json" % (camp, seed))
    if op.exists():
        print("already done", op); return
    Path("gurobi.env").write_text("Threads 1\n")
    gammas = [float(g) for g in gg]
    print("task %d -> %s seed %d | BINS=%d | arms=%s | gammas=%s"
          % (a.task_id, camp, seed, BINS, [a0 for a0, _, _ in ARMS], gammas), flush=True)
    t0 = time.time()
    jobs = [(seed, a.n, a.nte, ce, cp) for ce in CEPS for cp in CAPS]
    with mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                      initargs=(str((ROOT / dgp).resolve()), gammas)) as pool:
        res = pool.map(run_cell, jobs)
    op.write_text(json.dumps({"campaign": camp, "seed": seed, "N_train": a.n, "bins": BINS,
                              "gammas": ["%g" % g for g in gammas], "methods": METHODS,
                              "arms": [a0 for a0, _, _ in ARMS],
                              "cells": {k: v for k, v, _ in res},
                              "refs": {k: r for k, _, r in res}}))
    print("saved %s (%.1f min)" % (op, (time.time() - t0) / 60), flush=True)
    print("DONE_MARKER", flush=True)


if __name__ == "__main__":
    main()
