#!/usr/bin/env python3
"""X-X on KMZ with BOTH test and train values (plus per-seed curves) -- the train/test
toggle needs in-sample numbers and xxL_kmz.json stored only test means + seed-0 curves.
Protocol identical to run_xx_L.py (same draws, same solvers, same Shapley deployment);
the stored test means are re-derived as a cross-check."""
import sys, json, importlib.util
from pathlib import Path
import numpy as np

ROOT = "/home1/haghim/code 1.1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/extensions/Shapley")
import common
from shapley import extract_support

def LD(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return getattr(m, n)

d = importlib.util.module_from_spec(
    importlib.util.spec_from_file_location("dkm", ROOT + "/assets/exp_msmbench/dgp_g15.py"))
sys.modules["dkm"] = d
importlib.util.spec_from_file_location("dkm", ROOT + "/assets/exp_msmbench/dgp_g15.py").loader.exec_module(d)

S = {"IPW-X-X": (LD(ROOT + "/methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped"), "ipw"),
     "DoublyRobust-X-X": (LD(ROOT + "/methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped"), "dr"),
     "Direct-X-X": (LD(ROOT + "/methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py", "solve_direct_x_x_uncapped"), "direct")}
LGRID = [None, 3.0, 1.0]
LK = ["inf", "3", "1"]
PG = np.linspace(-1.0, 1.0, 41)


def shap(Xnew, sX, sp, block=200):
    Xnew = np.asarray(Xnew, float).reshape(-1, 1)
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


test_v = {m: {l: {} for l in LK} for m in S}
train_v = {m: {l: {} for l in LK} for m in S}
curves = {m: {l: {} for l in LK} for m in S}
for sd in range(5):
    obs, ftr = d.generate(400, sd)
    X, T, Y = obs["X"], obs["T"], obs["Y"]
    Y1tr, Y0tr = np.asarray(ftr["Y1"], float), np.asarray(ftr["Y0"], float)
    te, ft = d.generate(20000, sd + 1000)
    Xte = te["X"]; Y1, Y0 = ft["Y1"], ft["Y0"]
    w, _ = common.ipw_weights_from_data(X, T, 2)
    mu = common.outcome_means(X, T, Y, n_arms=2, cross_fit=True)
    for m, (fn, kind) in S.items():
        for L, lk in zip(LGRID, LK):
            if kind == "ipw":   r = fn(X, T, Y, w, n_arms=2, discretize=False, lipschitz=L)
            elif kind == "dr":  r = fn(X, T, Y, w, mu, n_arms=2, discretize=False, lipschitz=L)
            else:               r = fn(X, T, Y, n_arms=2, discretize=False, lipschitz=L)
            pi = np.asarray(r.pi[1], float)
            train_v[m][lk][str(sd)] = round(float(np.mean(pi * Y1tr + (1 - pi) * Y0tr)), 4)
            sX, spv = extract_support(X, r.pi[1])
            sX = np.asarray(sX, float).reshape(-1, 1); spv = np.asarray(spv, float).ravel()
            pe = shap(Xte.ravel(), sX, spv)
            test_v[m][lk][str(sd)] = round(float(np.mean(pe * Y1 + (1 - pe) * Y0)), 4)
            curves[m][lk][str(sd)] = [round(float(v), 3) for v in shap(PG, sX, spv)]
    print("seed %d done" % sd, flush=True)

out = {"n": 400, "seeds": 5, "Lgrid": LK, "policy_grid": [round(float(v), 4) for v in PG],
       "test": test_v, "train": train_v, "curves": curves,
       "test_mean": {m: {l: round(float(np.mean(list(v.values()))), 4)
                         for l, v in ll.items()} for m, ll in test_v.items()},
       "train_mean": {m: {l: round(float(np.mean(list(v.values()))), 4)
                          for l, v in ll.items()} for m, ll in train_v.items()}}
json.dump(out, open(ROOT + "/assets/grand/xx_km_train.json", "w"))
print("saved xx_km_train.json | test_mean:", out["test_mean"], flush=True)
print("train_mean:", out["train_mean"], flush=True)
