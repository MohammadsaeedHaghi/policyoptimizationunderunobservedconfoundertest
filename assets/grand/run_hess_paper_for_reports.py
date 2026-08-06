"""Hess et al., PAPER-EXACT (their github's own training recipe), at each report's protocol.

Replaces the custom-trainer numbers in hess_for_reports.json. Writes a NEW file so the old series
stays for the record. Pipeline per fit: disjoint 50/50 nuisance/policy split, {64,32} ReLU nets
for every nuisance and the policy, Adam lr 1e-3, batch 64, <=300 epochs, early stopping patience
10 -- github.com/konstantinhess/Efficient_sharp_policy_learning verbatim. The regularity flag is
carried over: gstar's outcome is bimodal given (x, a), which violates their Theorem 4.3 density
assumption, so its row stays marked untrusted.
"""
import sys, json, importlib.util
import numpy as np
ROOT = "/home1/haghim/code 1.1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/methods/SharpHess")
from sharp_hess import hess_paper, apply_hess_paper

def LD(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

JOBS = [("gs_cont", "assets/exp_gstar/dgp_cont.py", 400, 5, [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8], False),
        ("km",      "assets/exp_msmbench/dgp_g15.py", 400, 5, [1, 2, 3, 4.4817, 6, 8], True),
        ("kz",      "assets/exp_kz18/dgp.py", 200, 5, [1, 2, 3, 4.4817, 6, 8], True)]
out = {"note": "PAPER-EXACT: their repo's {64,32} nets + Adam recipe end to end; see sharp_hess.hess_paper"}
for tag, path, N, seeds, GG, trusted in JOBS:
    d = LD(ROOT + "/" + path, "hp_" + tag)
    te, ft = d.generate(200000, 999); xte = te["X"].ravel(); Y1, Y0 = ft["Y1"], ft["Y0"]
    means, sds = [], []
    for g in GG:
        vals = []
        for sd in range(seeds):
            obs, _ = d.generate(N, sd); X, T, Y = obs["X"], obs["T"], obs["Y"]
            pol = hess_paper(np.asarray(X, float).reshape(-1, 1), T, Y, float(g), seed=sd,
                             maximize=True)
            pe = apply_hess_paper(pol, xte.reshape(-1, 1))
            vals.append(float(np.mean(pe * Y1 + (1 - pe) * Y0)))
        means.append(round(float(np.mean(vals)), 4)); sds.append(round(float(np.std(vals)), 4))
        print("  %s G=%-8s %.3f +- %.3f" % (tag, g, means[-1], sds[-1]), flush=True)
    out[tag] = {"n": N, "seeds": seeds, "gammas": ["%g" % g for g in GG],
                "mean": means, "sd": sds, "regularity_ok": trusted}
json.dump(out, open(ROOT + "/assets/grand/hess_paper_for_reports.json", "w"), indent=1)
print("saved assets/grand/hess_paper_for_reports.json")
