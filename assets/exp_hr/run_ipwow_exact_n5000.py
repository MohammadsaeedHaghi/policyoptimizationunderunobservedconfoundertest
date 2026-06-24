"""IPW-O-W at TRAIN N=5000 — the EXACT library algorithm (per-unit Wasserstein transport, NO aggregation).
Uses the counting propensity (Hájek). epsilon=None so the solver computes the tight per-arm ε itself, exactly
as the method does. Reports ε_t (per arm) + the policy. This is O(n^2) and may exhaust 16 GB at n=5000.
Usage: python3 run_ipwow_exact_n5000.py"""
import sys, json, importlib.util, time, resource, traceback
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_hr"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("hrdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["hrdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
solve_ipw_o_w = L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")

def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)   # macOS bytes -> GB

K, N, SEED, GAMMA = 2, 5000, 0, 12.0
GRID = d.X_GRID
obs, _ = d.generate_data(N, SEED, "x_varying")
w, P = common.count_ipw_weights_from_data(obs["X"], obs["T"], K)        # counting propensity -> Hájek weights
print("n=%d  arm counts=%s  unique-X=%d  peakRSS=%.2fGB" % (N, np.bincount(obs["T"]).tolist(),
      len(np.unique(np.round(obs["X"], 2))), rss_gb()), flush=True)
print("building + solving EXACT per-unit IPW-O-W (Γ=%g, zscore=False)..." % GAMMA, flush=True)
t0 = time.time()
try:
    res = solve_ipw_o_w(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=GAMMA, discretize=False, zscore=False)
    dt = time.time() - t0
    xr = np.round(obs["X"].ravel(), 2)
    pol = [round(float(res.pi[1, np.where(xr == round(float(g), 2))[0][0]]), 4) for g in GRID]
    eps = tuple(float(e) for e in res.epsilon)
    print("DONE in %.1f min  peakRSS=%.2fGB" % (dt / 60, rss_gb()), flush=True)
    print("ε_t used (per arm): ε_0(control)=%.6f  ε_1(treat)=%.6f" % (eps[0], eps[1]), flush=True)
    print("treat-frac=%.3f   policy(grid)=%s" % (float(np.mean(pol)), pol), flush=True)
    (HERE / "hr_ipwow_exact_n5000.json").write_text(json.dumps(
        {"N_train": N, "Gamma": GAMMA, "epsilon": {"control": eps[0], "treat": eps[1]},
         "policy_grid": pol, "treat_frac": float(np.mean(pol)), "solve_minutes": dt / 60}, indent=2))
    print("saved hr_ipwow_exact_n5000.json", flush=True)
except MemoryError:
    print("OUT-OF-MEMORY after %.1f min (peakRSS=%.2fGB) — exact per-unit IPW-O-W does not fit at n=%d on 16GB." % ((time.time() - t0) / 60, rss_gb(), N), flush=True)
except Exception as e:
    print("FAILED after %.1f min (peakRSS=%.2fGB): %s" % ((time.time() - t0) / 60, rss_gb(), repr(e)[:300]), flush=True)
    traceback.print_exc()
