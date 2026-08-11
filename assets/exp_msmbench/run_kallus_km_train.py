#!/usr/bin/env python3
"""Kallus (paper port) on KMZ with BOTH test and train values, one gamma per task.
Protocol identical to exp_owgap_alpha/run_kallus.py --paper --eval draws (raw Y, raw
weights, test draw seed+1000, n_test 4000); the test values are cross-checked against
kmz_kallus*.json at aggregation time."""
import sys, json, argparse, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/home1/haghim/code 1.1")
sys.path.insert(0, str(ROOT))
import common

def LD(p, n):
    s = importlib.util.spec_from_file_location(n, str(p)); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

d = LD(ROOT / "assets/exp_msmbench/dgp_g15.py", "dkm")
kal = LD(ROOT / "methods/Kallus/kallus.py", "kal_mod")

GG = [1.0, 2.0, 3.0, 4.4817, 6.0, 8.0, 15.0, 50.0]
PG = np.linspace(-1.0, 1.0, 41)

ap = argparse.ArgumentParser()
ap.add_argument("--gamma-idx", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
g = GG[a.gamma_idx]

test_v, train_v, curves = {}, {}, {}
for sd in range(5):
    obs, ftr = d.generate(400, sd)
    te, ft = d.generate(4000, sd + 1000)
    wraw, _ = common.ipw_weights_from_data(obs["X"], obs["T"], 2, normalize=False)
    res = kal.fit_kallus_paper(obs["X"], obs["T"], obs["Y"], wraw, n_arms=2,
                               Gamma=g, maximize=True, seed=sd)
    pe = np.clip(kal.predict_kallus_paper(res.theta, np.asarray(te["X"], float).reshape(-1, 1))[:, 1], 0, 1)
    test_v[str(sd)] = round(float(np.mean(pe * ft["Y1"] + (1 - pe) * ft["Y0"])), 4)
    ptr = np.clip(kal.predict_kallus_paper(res.theta, np.asarray(obs["X"], float).reshape(-1, 1))[:, 1], 0, 1)
    train_v[str(sd)] = round(float(np.mean(ptr * np.asarray(ftr["Y1"], float)
                                           + (1 - ptr) * np.asarray(ftr["Y0"], float))), 4)
    pg = np.clip(kal.predict_kallus_paper(res.theta, PG.reshape(-1, 1))[:, 1], 0, 1)
    curves[str(sd)] = [round(float(v), 3) for v in pg]
    print("  G=%g seed %d test %.4f train %.4f" % (g, sd, test_v[str(sd)], train_v[str(sd)]),
          flush=True)

json.dump({"gamma": "%g" % g, "policy_grid": [round(float(v), 4) for v in PG],
           "test": test_v, "train": train_v, "curves": curves,
           "test_mean": round(float(np.mean(list(test_v.values()))), 4),
           "train_mean": round(float(np.mean(list(train_v.values()))), 4)},
          open(a.out, "w"))
print("saved %s" % a.out, flush=True)
