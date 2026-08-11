#!/usr/bin/env python3
"""Policy CURVES pi(x) for Hess (paper-exact) on KMZ -- the one series the report's new
policy-plot figure lacks. Protocol identical to run_hess_paper_for_reports.py's km row
(dgp_g15.generate, N=400, seeds 0-4, hess_paper(seed=sd)), extended to the full report
Gamma grid {1,2,3,4.4817,6,8,15,50}; per-seed test values are stored alongside so each fit
can be cross-checked against hess_paper_for_reports.json / hess_paper_km_g{15,50}.json.
Curves are evaluated on the SAME 41-point grid every other KMZ policy curve uses."""
import sys, json, argparse, importlib.util
import numpy as np

ROOT = "/home1/haghim/code 1.1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/methods/SharpHess")
from sharp_hess import hess_paper, apply_hess_paper

s = importlib.util.spec_from_file_location("dkm", ROOT + "/assets/exp_msmbench/dgp_g15.py")
d = importlib.util.module_from_spec(s); sys.modules["dkm"] = d; s.loader.exec_module(d)

GG = [1.0, 2.0, 3.0, 4.4817, 6.0, 8.0, 15.0, 50.0]
PG = np.linspace(-1.0, 1.0, 41)

ap = argparse.ArgumentParser()
ap.add_argument("--gamma-idx", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
g = GG[a.gamma_idx]

te, ft = d.generate(200000, 999)
xte = te["X"].ravel(); Y1, Y0 = ft["Y1"], ft["Y0"]

curves, values, values_tr = {}, [], []
for sd in range(5):
    obs, ftr = d.generate(400, sd)
    X, T, Y = obs["X"], obs["T"], obs["Y"]
    pol = hess_paper(np.asarray(X, float).reshape(-1, 1), T, Y, float(g), seed=sd,
                     maximize=True)
    pe = apply_hess_paper(pol, xte.reshape(-1, 1))
    values.append(float(np.mean(pe * Y1 + (1 - pe) * Y0)))
    ptr = np.clip(apply_hess_paper(pol, np.asarray(X, float).reshape(-1, 1)), 0.0, 1.0)
    values_tr.append(float(np.mean(ptr * np.asarray(ftr["Y1"], float)
                                   + (1 - ptr) * np.asarray(ftr["Y0"], float))))
    pg = apply_hess_paper(pol, PG.reshape(-1, 1))
    curves[str(sd)] = [round(float(v), 3) for v in np.clip(pg, 0.0, 1.0)]
    print("  G=%g seed %d test %.4f train %.4f" % (g, sd, values[-1], values_tr[-1]),
          flush=True)

json.dump({"gamma": "%g" % g, "policy_grid": [round(float(v), 4) for v in PG],
           "curves": curves, "values": values, "values_train": values_tr,
           "mean": round(float(np.mean(values)), 4),
           "mean_train": round(float(np.mean(values_tr)), 4)}, open(a.out, "w"))
print("saved %s (test %.4f train %.4f)"
      % (a.out, float(np.mean(values)), float(np.mean(values_tr))), flush=True)
