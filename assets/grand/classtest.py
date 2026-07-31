"""Is O-W's deficit vs sharp the UNCERTAINTY SET or the POLICY CLASS?

Sharp is piecewise-constant over ~15 cells (~15 free parameters). Every campaign so far ran our
solvers with discretize=False -- a free pi per support point, i.e. up to n parameters. This holds
everything else fixed and gives O-W sharp's policy class via discretize=True + mesh, so the only
remaining difference is the uncertainty set itself.

KMZ uncapped, n=500, matched Gamma=4.4817. Sharp's number on this exact cell: 1.216 (10 seeds).
"""
import sys, importlib.util
import numpy as np
ROOT = "/home1/haghim/code 1.1"
sys.path.insert(0, ROOT); sys.path.insert(0, ROOT + "/extensions/Shapley"); sys.path.insert(0, ROOT + "/assets/grand")
import common
from shapley import extract_support
from capped_refs import _shapley_fast

def LD(rel, fn):
    s = importlib.util.spec_from_file_location(fn, ROOT + "/" + rel)
    m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m)
    return getattr(m, fn)

OW = LD("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
OX = LD("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
sp = importlib.util.spec_from_file_location("d", ROOT + "/assets/exp_msmbench/dgp_g15.py")
d = importlib.util.module_from_spec(sp); sys.modules["d"] = d; sp.loader.exec_module(d)
N, NTE, G = 500, 4000, 4.4817
acc = {}
for sd in range(6):
    obs, _ = d.generate(N, sd); X, T, Y = obs["X"], obs["T"], obs["Y"]
    te, ft = d.generate(NTE, sd + 1000); Xte = te["X"]; Y1, Y0 = ft["Y1"], ft["Y0"]
    w, _ = common.ipw_weights_from_data(X, T, 2)
    Dm = common.pairwise_distance_matrix(X)
    eps = tuple(common.tight_epsilon(Dm, T, w, 2, is_distance=True, c_eps=1.0))
    def ev(r):
        sX, spv = extract_support(X, r.pi[1])
        pe = _shapley_fast(np.asarray(Xte, float), np.asarray(sX, float), np.asarray(spv, float).ravel())
        return float(np.mean(pe * Y1 + (1 - pe) * Y0))
    for tag, kw in (("O-W free pi (all campaigns)", dict(discretize=False, lipschitz=3.0)),
                    ("O-W mesh=10", dict(discretize=True, mesh=10)),
                    ("O-W mesh=15 (sharp's class)", dict(discretize=True, mesh=15)),
                    ("O-W mesh=25", dict(discretize=True, mesh=25))):
        try: acc.setdefault(tag, []).append(ev(OW(X, T, Y, w, n_arms=2, Gamma=G, zscore=False, epsilon=eps, **kw)))
        except Exception as e: print("FAIL", tag, e, flush=True)
    for tag, kw in (("O-X free pi", dict(discretize=False, lipschitz=3.0)),
                    ("O-X mesh=15", dict(discretize=True, mesh=15))):
        try: acc.setdefault(tag, []).append(ev(OX(X, T, Y, w, n_arms=2, Gamma=G, **kw)))
        except Exception as e: print("FAIL", tag, e, flush=True)
    print("seed", sd, "done", flush=True)
print("\n=== KMZ uncapped n=500, matched Gamma, %d seeds ===" % len(acc[list(acc)[0]]))
for k, v in sorted(acc.items(), key=lambda kv: -np.mean(kv[1])):
    print("  %-30s %.3f +- %.3f" % (k, np.mean(v), np.std(v)))
print("  %-30s %.3f   (same protocol, 10 seeds)" % ("Sharp-O-X", 1.216))
print("CLASSTEST_DONE")
