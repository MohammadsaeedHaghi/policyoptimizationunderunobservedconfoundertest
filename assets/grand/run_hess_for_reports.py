"""Hess et al. (2502.13022) run at each report's OWN protocol, for the report tables.

Also records the regularity diagnostic that decides whether the number is trustworthy on that
DGP: their Theorem 4.3 assumes p(y|x,a) has a density bounded away from zero near F^-1(alpha+).
gstar's outcome is bimodal given (x,a) (the confounder shifts levels by +-8), which violates it,
and there the estimator is ANTI-correlated with the bound it estimates (rank-corr -0.41 against
the analytically computed truth, vs +0.69 for the plug-in). We store that flag alongside.
"""
import sys, json, importlib.util
import numpy as np
ROOT = "/home1/haghim/code 1.1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/methods/SharpHess")
from sharp_hess import fit_scores, learn_policy_parametric, apply_policy

def LD(p, n):
    s = importlib.util.spec_from_file_location(n, p); m = importlib.util.module_from_spec(s)
    sys.modules[n] = m; s.loader.exec_module(m); return m

JOBS = [("gs_cont", "assets/exp_gstar/dgp_cont.py", 400, 5, [1,1.5,2,2.5,3,4,5,6,8], False),
        ("km",      "assets/exp_msmbench/dgp_g15.py", 400, 5, [1,2,3,4.4817,6,8], True),
        ("kz",      "assets/exp_kz18/dgp.py", 200, 5, [1,2,3,4.4817,6,8], True)]
out = {"note": "Hess et al. arXiv 2502.13022, Eq.15 + Algorithm 1, outcome standardised as in their data_gen.py"}
for tag, path, N, seeds, GG, trusted in JOBS:
    d = LD(ROOT + "/" + path, "hr_" + tag)
    te, ft = d.generate(200000, 999); xte = te["X"].ravel(); Y1, Y0 = ft["Y1"], ft["Y0"]
    means, sds = [], []
    for g in GG:
        vals = []
        for sd in range(seeds):
            obs, _ = d.generate(N, sd); X, T, Y = obs["X"], obs["T"], obs["Y"]
            sc = fit_scores(X, T, Y, float(g), k=max(40, N // 10), n_folds=2, maximize=True,
                            seed=sd, standardize=True)
            pol = learn_policy_parametric(X, sc, n_iter=600, lr=0.3, restarts=5, seed=sd,
                                          maximize=True, hidden=8)
            pe = apply_policy(pol, xte.reshape(-1, 1))
            vals.append(float(np.mean(pe * Y1 + (1 - pe) * Y0)))
        means.append(round(float(np.mean(vals)), 4)); sds.append(round(float(np.std(vals)), 4))
        print("  %s G=%-8s %.3f +- %.3f" % (tag, g, means[-1], sds[-1]), flush=True)
    out[tag] = {"n": N, "seeds": seeds, "gammas": ["%g" % g for g in GG],
                "mean": means, "sd": sds, "regularity_ok": trusted}
json.dump(out, open(ROOT + "/assets/grand/hess_for_reports.json", "w"), indent=1)
print("saved assets/grand/hess_for_reports.json")
