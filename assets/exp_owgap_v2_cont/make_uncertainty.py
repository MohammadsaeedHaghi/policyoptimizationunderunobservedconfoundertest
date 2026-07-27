#!/usr/bin/env python3
"""Per-seed values + paired CIs for the continuous v2 results (license-free post-processing).

The Shapley result files persist the RAW per-seed support policies (policies_support_by_seed),
so per-seed test values are recomputable without any solver: regenerate the deterministic
training/test draws, re-deploy each saved support policy with the same Shapley operator, and
evaluate on the known potential outcomes. Produces, per transport budget:
  - per-seed values for IPW-O-W / DoublyRobust-O-W at the G=4,L=3 cell and at each method's
    best cell, and for the naive DR baseline,
  - mean +- SD, and the PAIRED per-seed margin vs naive with a 95% t CI (df = n_seeds - 1).

Writes assets/exp_owgap_v2_cont/cont_uncertainty.json. Validates that recomputed per-seed
means reproduce the stored surface cells.
"""
import sys, json, importlib.util
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "extensions" / "Shapley"))
from shapley import extract_support

def _load(p, n):
    s = importlib.util.spec_from_file_location(n, str(p)); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

runner = _load(ROOT / "assets" / "run_owgap_lip_gamma_2d.py", "lg2d_runner")
d = _load(HERE / "dgp.py", "v2cont_dgp")

FILES = {"1.0": "owgap_v2_lip_gamma_2d_shapley.json",
         "1.5": "owgap_v2_lip_gamma_2d_shapley_ce1.5.json",
         "2.0": "owgap_v2_lip_gamma_2d_shapley_ce2.0.json"}
TCRIT = {7: 2.365, 19: 2.093}

def deploy_value(pi_sup, X, Xte, Y1t, Y0t):
    sX, sp = extract_support(X.reshape(-1, 1), np.asarray(pi_sup, float))
    pe = runner._shapley_fast(Xte.reshape(-1, 1), np.asarray(sX, float), np.asarray(sp, float).ravel())
    return float(np.mean(pe * Y1t + (1 - pe) * Y0t))

out = {}
for ce, fn in FILES.items():
    j = json.load(open(HERE / fn))
    sup = j["policies_support_by_seed"]; seeds = sorted(int(s) for s in sup)
    cells = {"IPW-O-W": [("4", "3"), (j["best"]["IPW-O-W"]["gamma"], j["best"]["IPW-O-W"]["L"])],
             "DoublyRobust-O-W": [("4", "3"),
                                  (j["best"]["DoublyRobust-O-W"]["gamma"], j["best"]["DoublyRobust-O-W"]["L"])]}
    per = {"naive": []}
    for m, cl in cells.items():
        for g, l in cl: per[f"{m}@G{g},L{l}"] = []
    for sd in seeds:
        obs, _ = d.generate(j["N_train"], sd)
        te, ft = d.generate(j["N_test"], sd + 1000)
        X, Xte, Y1t, Y0t = obs["X"].ravel(), te["X"].ravel(), ft["Y1"], ft["Y0"]
        s = sup[str(sd)]
        per["naive"].append(deploy_value(s["_naive_dr"], X, Xte, Y1t, Y0t))
        for m, cl in cells.items():
            for g, l in dict.fromkeys(cl):
                per[f"{m}@G{g},L{l}"].append(deploy_value(s[m][g][l], X, Xte, Y1t, Y0t))
    n = len(seeds); tc = TCRIT.get(n - 1, 2.365)
    res = {"n_seeds": n, "per_seed": {k: [round(v, 4) for v in vs] for k, vs in per.items()}, "cells": {}}
    for k, vs in per.items():
        a = np.array(vs)
        e = {"mean": round(a.mean(), 4), "sd": round(a.std(ddof=1), 4)}
        if k != "naive":
            dif = a - np.array(per["naive"])
            e["margin_mean"] = round(dif.mean(), 4)
            e["margin_ci95"] = round(tc * dif.std(ddof=1) / np.sqrt(n), 4)
        res["cells"][k] = e
    # validation vs stored surface (means were rounded to 4dp there)
    chk = abs(res["cells"]["IPW-O-W@G4,L3"]["mean"] - j["surface"]["IPW-O-W"]["4"]["3"])
    chk2 = abs(res["cells"]["naive"]["mean"] - j["naive_dr"])
    res["validation_absdiff"] = {"ipwow_G4L3": round(chk, 4), "naive": round(chk2, 4)}
    out[ce] = res
    print("ce=%s  naive %.3f+-%.2f  IPW-O-W@G4L3 %.3f+-%.2f  margin %+.3f +- %.3f  (check %.4f/%.4f)" %
          (ce, res["cells"]["naive"]["mean"], res["cells"]["naive"]["sd"],
           res["cells"]["IPW-O-W@G4,L3"]["mean"], res["cells"]["IPW-O-W@G4,L3"]["sd"],
           res["cells"]["IPW-O-W@G4,L3"]["margin_mean"], res["cells"]["IPW-O-W@G4,L3"]["margin_ci95"],
           chk, chk2), flush=True)

(HERE / "cont_uncertainty.json").write_text(json.dumps(out, indent=1))
print("saved cont_uncertainty.json")
