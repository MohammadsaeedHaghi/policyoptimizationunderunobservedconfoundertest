"""Realised test outcome vs Γ for every method on the HR x_varying DGP.
Train each method over a Γ grid, deploy on a fresh N_test=5000 test set, average the realised outcome
(using the test units' true potential outcomes). Γ-free methods (IPW-X-X, Direct-X-X, Oracle) are flat lines.
All methods use the COUNTING propensity. The O(n²) Wasserstein methods (IPW-O-W, Hajek-O-W) are trained at
N=600 (can't reach 5000); the fast/box methods at N=5000. Builds var HR_VALUE_GAMMA + splices into index.html.
Usage: python3 run_value_vs_gamma.py"""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_hr"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("hrdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["hrdgp"] = d; sp.loader.exec_module(d)
def L(rel, fn):
    s = importlib.util.spec_from_file_location(fn, str(ROOT / rel)); m = importlib.util.module_from_spec(s); sys.modules[fn] = m; s.loader.exec_module(m); return getattr(m, fn)
ipwxx = L("methods/IPW-X-X/Uncapped/ipw_x_x_uncapped.py", "solve_ipw_x_x_uncapped")
directxx = L("methods/Direct-X-X/Uncapped/direct_x_x_uncapped.py", "solve_direct_x_x_uncapped")
ipwox = L("methods/IPW-O-X/Uncapped/ipw_o_x_uncapped.py", "solve_ipw_o_x_uncapped")
hajekox = L("methods/Hajek-O-X/Uncapped/hajek_o_x_uncapped.py", "solve_hajek_o_x_uncapped")
ipwow = L("methods/IPW-O-W/Uncapped/ipw_o_w_uncapped.py", "solve_ipw_o_w_uncapped")
hajekow = L("methods/Hajek-O-W/Uncapped/hajek_o_w_uncapped.py", "solve_hajek_o_w_uncapped")

K, N_TR, N_W, N_TEST = 2, 5000, 600, 5000
X = d.X_GRID
GAMMAS = [1, 2, 3, 4, 6, 8, 12]

# ---- train sets ----
obs5, _ = d.generate_data(N_TR, 0, "x_varying")
w5, P5 = common.count_ipw_weights_from_data(obs5["X"], obs5["T"], K)         # Hájek (IPW-X-X, IPW-O-X, ...)
wraw5 = 1.0 / common.factual_propensity(P5, obs5["T"])                       # raw (Hajek-O-X)
xr5 = np.round(obs5["X"], 2)
obsW, _ = d.generate_data(N_W, 0, "x_varying")
wW, _ = common.count_ipw_weights_from_data(obsW["X"], obsW["T"], K)
xrW = np.round(obsW["X"], 2)

def grid_of(pi, xr): return np.array([float(pi[1, np.where(xr == round(float(gx), 2))[0][0]]) for gx in X])

# ---- test set + realised-value operator ----
_, tf = d.generate_data(N_TEST, 1, "x_varying")
txr = np.round(tf["X"], 2); Y1t = tf["Y1"].astype(float); Y0t = tf["Y0"].astype(float)
_pos = {round(float(X[i]), 2): i for i in range(len(X))}
def test_realised(pol_grid):
    pit = np.array([pol_grid[_pos[round(float(x), 2)]] for x in txr])
    return float(np.mean(pit * Y1t + (1.0 - pit) * Y0t))

# ---- policies ----
pol_ipw = grid_of(ipwxx(obs5["X"], obs5["T"], obs5["Y"], w5, n_arms=K, discretize=False).pi, xr5)
pol_dir = grid_of(directxx(obs5["X"], obs5["T"], obs5["Y"], n_arms=K, discretize=False).pi, xr5)
oracle = d.oracle_policy(X)
ox, hox, iow, how = {}, {}, {}, {}
for g in GAMMAS:
    ox[g] = grid_of(ipwox(obs5["X"], obs5["T"], obs5["Y"], w5, n_arms=K, Gamma=float(g), discretize=False).pi, xr5)
    hox[g] = grid_of(hajekox(obs5["X"], obs5["T"], obs5["Y"], wraw5, n_arms=K, Gamma=float(g), maximize=True, discretize=False).pi, xr5)
    iow[g] = grid_of(ipwow(obsW["X"], obsW["T"], obsW["Y"], wW, n_arms=K, Gamma=float(g), discretize=False, zscore=False).pi, xrW)
    how[g] = grid_of(hajekow(obsW["X"], obsW["T"], obsW["Y"], wW, n_arms=K, Gamma=float(g), maximize=True, discretize=False, zscore=False).pi, xrW)
    print("Γ=%-2d  IPW-O-X=%.3f Hajek-O-X=%.3f IPW-O-W=%.3f Hajek-O-W=%.3f" % (
        g, test_realised(ox[g]), test_realised(hox[g]), test_realised(iow[g]), test_realised(how[g])), flush=True)

V = lambda d_: [round(test_realised(d_[g]), 4) for g in GAMMAS]
flat = lambda p: [round(test_realised(p), 4)] * len(GAMMAS)
curves = {"IPW-X-X": flat(pol_ipw), "Direct-X-X": flat(pol_dir), "Oracle": flat(oracle),
          "IPW-O-X": V(ox), "Hajek-O-X": V(hox), "IPW-O-W": V(iow), "Hajek-O-W": V(how)}
print("FLAT  IPW-X-X=%.3f Direct-X-X=%.3f Oracle=%.3f" % (curves["IPW-X-X"][0], curves["Direct-X-X"][0], curves["Oracle"][0]))

def ser(idl, color, y): return {"id": idl, "label": idl, "color": color, "y": y}
HV = {"x": [float(g) for g in GAMMAS], "xmin": 1.0, "xmax": float(GAMMAS[-1]), "xlabel": "Γ  (sensitivity-model strength)",
      "ylabel": "realised E[Y] on test (N=5000)", "ymin": -0.06, "ymax": 0.10,
      "series": [ser("IPW-X-X", "#1f77b4", curves["IPW-X-X"]), ser("Direct-X-X", "#ff7f0e", curves["Direct-X-X"]),
                 ser("IPW-O-X", "#1f77b4", curves["IPW-O-X"]), ser("Hajek-O-X", "#9467bd", curves["Hajek-O-X"]),
                 ser("IPW-O-W", "#1f77b4", curves["IPW-O-W"]), ser("Hajek-O-W", "#9467bd", curves["Hajek-O-W"]),
                 ser("Oracle", "#444444", curves["Oracle"])],
      "hlines": [{"y": curves["Oracle"][0], "color": "#94a3b8", "dash": "4,4", "label": "oracle ceiling"}],
      "note": "Realised test E[Y] vs Γ (N_test=5000). Flat: IPW-X-X/Direct-X-X/Oracle. Box-robust (IPW-O-X/Hajek-O-X) trained N=5000; Wasserstein (IPW-O-W/Hajek-O-W) trained N=600 (O(n²))."}
(HERE / "hr_value_vs_gamma.json").write_text(json.dumps({"gammas": GAMMAS, "curves": curves}, indent=2))

# PNG
fig, ax = plt.subplots(figsize=(7.8, 4.6))
STY = {"IPW-X-X": ("#1f77b4", "--", ""), "Direct-X-X": ("#ff7f0e", "--", ""), "Oracle": ("#444444", ":", ""),
       "IPW-O-X": ("#1f77b4", "-", "s"), "Hajek-O-X": ("#9467bd", "-", "s"),
       "IPW-O-W": ("#1f77b4", "-", "o"), "Hajek-O-W": ("#9467bd", "-", "o")}
for nm, y in curves.items():
    c, ls, mk = STY[nm]; ax.plot(GAMMAS, y, color=c, ls=ls, marker=mk, ms=5, lw=2.2, label=nm)
ax.set_xlabel("Γ"); ax.set_ylabel("realised E[Y] on test (N=5000)"); ax.grid(alpha=0.25)
ax.legend(fontsize=7.5, ncol=2); ax.set_title("HR x_varying — realised test outcome vs Γ", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "value_vs_gamma.png", dpi=130); plt.close(fig)

# splice
idx = ROOT / "index.html"; h = idx.read_text()
line = "var HR_VALUE_GAMMA = %s;" % json.dumps(HV, separators=(",", ":"))
if re.search(r'^var HR_VALUE_GAMMA = .*;$', h, flags=re.M):
    h = re.sub(r'^var HR_VALUE_GAMMA = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var HR_POLICY = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
call = "renderChart('ichart-hrfast-vgamma',HR_VALUE_GAMMA);"
if call not in h:
    h = h.replace("renderChart('ichart-hrfast-policy',HR_POLICY);",
                  "renderChart('ichart-hrfast-policy',HR_POLICY);\n" + call, 1)
idx.write_text(h)
print("spliced HR_VALUE_GAMMA + renderChart call into index.html")
