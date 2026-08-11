#!/usr/bin/env python3
"""TRAIN-set values for the KMZ O-method surface -- no re-solving needed.

policies_support_by_seed in each kmz_main_ce*.json stores pi on ALL 400 training points
(plus the training X under "_X" and the naive plug-in under "_naive_dr"). The training
draw is reproducible (dgp_g15.generate(400, sd) -- asserted against the stored _X), and
the DGP returns both potential outcomes for the drawn rows, so the in-sample value of
every stored policy is a dot product:  V_tr = mean(pi * Y1_tr + (1-pi) * Y0_tr).

Output: kmz_train_values.json
  refs_by_seed[sd]  = {oracle, all, never}         (train draw)
  surface[ce][m][g][L] = mean over seeds of V_tr;  per_seed[...] keeps the seed values
  naive[sd], naive_mean
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np

ROOT = Path("/home1/haghim/code 1.1")
HERE = ROOT / "assets" / "exp_msmbench"
sp = importlib.util.spec_from_file_location("dkm", str(HERE / "dgp_g15.py"))
d = importlib.util.module_from_spec(sp); sys.modules["dkm"] = d; sp.loader.exec_module(d)

CES = ["1.0", "1.5", "2.0"]
files = {ce: json.loads((HERE / ("kmz_main_ce%s.json" % ce)).read_text()) for ce in CES}
seeds = sorted(files["1.0"]["policies_support_by_seed"].keys())

tr = {}
for sd in seeds:
    obs, full = d.generate(400, int(sd))
    X = np.asarray(obs["X"], float).ravel()
    stored = np.asarray(files["1.0"]["policies_support_by_seed"][sd]["_X"], float).ravel()
    assert np.max(np.abs(np.sort(X) - np.sort(stored))) < 1e-3, "train draw mismatch seed " + sd
    if np.max(np.abs(X - stored)) < 1e-3:        # same ordering (the runner stored X as drawn)
        Y1 = np.asarray(full["Y1"], float); Y0 = np.asarray(full["Y0"], float)
    else:                                        # realign outcomes to the stored ordering
        order = np.argsort(X); sorder = np.argsort(stored)
        Y1 = np.empty(len(X)); Y0 = np.empty(len(X))
        Y1[sorder] = np.asarray(full["Y1"], float)[order]
        Y0[sorder] = np.asarray(full["Y0"], float)[order]
    po = d.oracle_policy(stored)
    tr[sd] = {"Y1": Y1, "Y0": Y0,
              "refs": {"oracle": float(np.mean(po * Y1 + (1 - po) * Y0)),
                       "all": float(np.mean(Y1)), "never": float(np.mean(Y0))}}

out = {"refs_by_seed": {sd: tr[sd]["refs"] for sd in seeds},
       "surface": {}, "per_seed": {}, "naive": {}}
val = lambda sd, pi: float(np.mean(np.asarray(pi, float) * tr[sd]["Y1"]
                                   + (1 - np.asarray(pi, float)) * tr[sd]["Y0"]))
for ce in CES:
    sup = files[ce]["policies_support_by_seed"]
    out["surface"][ce] = {}; out["per_seed"][ce] = {}
    for m in files[ce]["methods"]:
        out["surface"][ce][m] = {}; out["per_seed"][ce][m] = {}
        for g in files[ce]["gammas"]:
            gd, gp = {}, {}
            for L in files[ce]["Lgrid"]:
                vs = {sd: round(val(sd, sup[sd][m][g][L]), 4) for sd in seeds
                      if g in sup[sd][m] and L in sup[sd][m][g]}
                if vs:
                    gd[L] = round(float(np.mean(list(vs.values()))), 4); gp[L] = vs
            if gd:
                out["surface"][ce][m][g] = gd; out["per_seed"][ce][m][g] = gp
for sd in seeds:
    out["naive"][sd] = round(val(sd, files["1.0"]["policies_support_by_seed"][sd]["_naive_dr"]), 4)
out["naive_mean"] = round(float(np.mean(list(out["naive"].values()))), 4)

(HERE / "kmz_train_values.json").write_text(json.dumps(out))
print("saved kmz_train_values.json | seeds", seeds,
      "| e.g. IPW-O-W ce1 g8 L3: train %.3f" % out["surface"]["1.0"]["IPW-O-W"]["8"]["3"],
      "| refs seed0:", {k: round(v, 3) for k, v in out["refs_by_seed"]["0"]["refs"].items()}
      if "refs" in out["refs_by_seed"]["0"] else out["refs_by_seed"]["0"])
