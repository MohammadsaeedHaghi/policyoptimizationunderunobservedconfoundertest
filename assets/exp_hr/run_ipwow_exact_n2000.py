"""EXACT per-unit IPW-O-W (library solver, NO aggregation) at TRAIN N=2000 on HR x_varying, for Γ=3,6,12.
Counting propensity (Hájek). ε computed once (Γ-independent) and reused. Evaluates on N_test=5000 and adds
IPW-O-W to the value-vs-Γ (HR_N2000) panel and the π policy plot (HR_POLICY) at those three Γ. Reports ε_t.
~25 min/solve. Usage: python3 run_ipwow_exact_n2000.py"""
import sys, json, re, importlib.util, time, resource
from pathlib import Path
import numpy as np
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_hr"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("hrdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["hrdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
solve_ipw_o_w = L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
def rss_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)

K, N = 2, 2000
GRID = d.X_GRID
GAMMAS_RUN = [3, 6, 12]
GAMMAS_GRID = [1, 2, 3, 4, 6, 8, 12]   # x-axis of the value-vs-Γ panel
obs, _ = d.generate_data(N, 0, "x_varying")
w, P = common.count_ipw_weights_from_data(obs["X"], obs["T"], K)        # counting -> Hájek weights
xr = np.round(obs["X"].ravel(), 2)
def grid_of(pi): return [round(float(pi[1, np.where(xr == round(float(g), 2))[0][0]]), 4) for g in GRID]

# ε once (Γ-independent), exactly as the method computes it (per-unit transport)
print("computing tight ε at N=2000 (per-unit transport)...", flush=True)
t0 = time.time()
D, _, _ = common.distance_matrix(np.asarray(obs["X"], float).reshape(-1, 1), zscore=False)
eps = common.tight_epsilon(D, obs["T"], w, K, is_distance=True, c_eps=1.0)
print("ε_t (per arm, c_eps=1.0): ε_0(control)=%.6f  ε_1(treat)=%.6f   (%.1f min, peakRSS=%.1fGB)" % (eps[0], eps[1], (time.time() - t0) / 60, rss_gb()), flush=True)

# test set + realised-value operator
_, tf = d.generate_data(5000, 1, "x_varying")
txr = np.round(tf["X"], 2); Y1t = tf["Y1"].astype(float); Y0t = tf["Y0"].astype(float)
_pos = {round(float(GRID[i]), 2): i for i in range(len(GRID))}
def test_realised(pg): pit = np.array([pg[_pos[round(float(x), 2)]] for x in txr]); return float(np.mean(pit * Y1t + (1 - pit) * Y0t))

pol_by_g, rv_by_g = {}, {}
for G in GAMMAS_RUN:
    t0 = time.time()
    res = solve_ipw_o_w(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(G), discretize=False, zscore=False, epsilon=eps)
    pol = grid_of(res.pi); pol_by_g[G] = pol; rv_by_g[G] = test_realised(pol)
    print("Γ=%-2d  treat-frac=%.2f  realised test E[Y]=%.4f   (%.1f min, peakRSS=%.1fGB)" % (G, float(np.mean(pol)), rv_by_g[G], (time.time() - t0) / 60, rss_gb()), flush=True)

out = {"epsilon": {"control": eps[0], "treat": eps[1]}, "gammas": GAMMAS_RUN, "N_train": N, "N_test": 5000,
       "policy_by_gamma": pol_by_g, "realised_test_by_gamma": rv_by_g}
(HERE / "hr_ipwow_exact_n2000.json").write_text(json.dumps(out, indent=2))
print("\nsaved hr_ipwow_exact_n2000.json", flush=True)

# ---- patch the two charts: add IPW-O-W (exact, N=2000) at Γ=3,6,12 ----
idx = ROOT / "index.html"; h = idx.read_text()
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)

# (a) value-vs-Γ panel HR_N2000: add an IPW-O-W series with values only at 3,6,12 (null elsewhere)
m = re.search(r'var HR_N2000 = (.*?);\n', h); HN = json.loads(m.group(1))
yv = [rv_by_g[int(g)] if int(g) in GAMMAS_RUN else None for g in GAMMAS_GRID]
HN["series"] = [s for s in HN["series"] if s["id"] != "IPW-O-W"]
HN["series"].insert(-1, {"id": "IPW-O-W", "label": "IPW-O-W (exact, N=2000)", "color": "#1f77b4", "y": yv})
HN["note"] = "Realised test E[Y] vs Γ, all box/DR methods at N_train=2000; the EXACT per-unit IPW-O-W (blue circles) added at Γ=3,6,12. N_test=5000."
h = h[:m.start()] + "var HR_N2000 = " + json.dumps(HN, separators=(",", ":")) + ";\n" + h[m.end():]

# (b) π policy plot HR_POLICY: add IPW-O-W policy at Γ=3,6,12 (null grid elsewhere)
m = re.search(r'var HR_POLICY = (.*?);\n', h); HP = json.loads(m.group(1))
nullgrid = [None] * len(GRID)
for gkey, series in HP["seriesByGamma"].items():
    series[:] = [s for s in series if s["id"] != "IPW-O-W"]
    gval = int(round(float(gkey)))
    yv = pol_by_g[gval] if gval in GAMMAS_RUN else nullgrid
    series.insert(-1, {"id": "IPW-O-W", "label": "IPW-O-W (exact, N=2000)", "color": "#1f77b4", "y": yv})
h = h[:m.start()] + "var HR_POLICY = " + json.dumps(HP, separators=(",", ":")) + ";\n" + h[m.end():]
idx.write_text(h)
print("patched HR_N2000 + HR_POLICY with exact IPW-O-W at Γ=3,6,12:", {g: round(rv_by_g[g], 4) for g in GAMMAS_RUN}, flush=True)
