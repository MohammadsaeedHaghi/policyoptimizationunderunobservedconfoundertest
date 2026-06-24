"""Re-run the box/fast methods at TRAIN N=2000 on HR x_varying, evaluate on the common N_test=5000, and
compare to the N=5000 numbers — to judge whether N=2000 is good enough (so the O(n^2) Wasserstein methods
can be run at the same N for a fair comparison). Methods: IPW-X-X, Direct-X-X, DoublyRobust-X-X (Γ-free) +
IPW-O-X, Hajek-O-X, DoublyRobust-O-X (Γ-swept). All use the COUNTING propensity. Usage: python3 run_methods_n2000.py"""
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
drxx = L("methods/DoublyRobust-X-X/Uncapped/doublyrobust_x_x_uncapped.py", "solve_doublyrobust_x_x_uncapped")
drox = L("methods/DoublyRobust-O-X/Uncapped/doublyrobust_o_x_uncapped.py", "solve_doublyrobust_o_x_uncapped")

K, N_TR = 2, 2000
GRID = d.X_GRID
GAMMAS = [1, 2, 3, 4, 6, 8, 12]
obs, _ = d.generate_data(N_TR, 0, "x_varying")
w, P = common.count_ipw_weights_from_data(obs["X"], obs["T"], K)        # Hájek counting weights
wraw = 1.0 / common.factual_propensity(P, obs["T"])                     # raw (Hajek-O-X)
muhat = common.outcome_means(obs["X"], obs["T"], obs["Y"], K)          # μ̂ (DR)
xr = np.round(obs["X"].ravel(), 2)
def grid_of(pi): return np.array([float(pi[1, np.where(xr == round(float(g), 2))[0][0]]) for g in GRID])

# test set + realised-value operator
_, tf = d.generate_data(5000, 1, "x_varying")
txr = np.round(tf["X"], 2); Y1t = tf["Y1"].astype(float); Y0t = tf["Y0"].astype(float)
_pos = {round(float(GRID[i]), 2): i for i in range(len(GRID))}
def test_realised(pg): pit = np.array([pg[_pos[round(float(x), 2)]] for x in txr]); return float(np.mean(pit * Y1t + (1 - pit) * Y0t))

# Γ-free policies
pol_ipw = grid_of(ipwxx(obs["X"], obs["T"], obs["Y"], w, n_arms=K, discretize=False).pi)
pol_dir = grid_of(directxx(obs["X"], obs["T"], obs["Y"], n_arms=K, discretize=False).pi)
pol_drx = grid_of(drxx(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, discretize=False).pi)
oracle = d.oracle_policy(GRID)
# Γ-swept
ox, hox, dro = {}, {}, {}
for g in GAMMAS:
    ox[g] = grid_of(ipwox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), discretize=False).pi)
    hox[g] = grid_of(hajekox(obs["X"], obs["T"], obs["Y"], wraw, n_arms=K, Gamma=float(g), maximize=True, discretize=False).pi)
    dro[g] = grid_of(drox(obs["X"], obs["T"], obs["Y"], w, muhat, n_arms=K, Gamma=float(g), discretize=False).pi)

V = lambda dd: [round(test_realised(dd[g]), 4) for g in GAMMAS]
flat = lambda p: [round(test_realised(p), 4)] * len(GAMMAS)
curves = {"IPW-X-X": flat(pol_ipw), "Direct-X-X": flat(pol_dir), "DoublyRobust-X-X": flat(pol_drx),
          "IPW-O-X": V(ox), "Hajek-O-X": V(hox), "DoublyRobust-O-X": V(dro), "Oracle": flat(oracle)}
print("=== realised test E[Y] @ N_train=2000 (N_test=5000) ===")
for k, y in curves.items():
    print("  %-16s best=%+.3f  byΓ=%s" % (k, max(y), [("%+.3f" % v) for v in y]))

# compare to the N=5000 run (box methods) if available
cmp = {}
p5 = HERE / "hr_value_vs_gamma.json"
if p5.exists():
    c5 = json.load(open(p5))["curves"]
    print("\n=== N=2000 vs N=5000 (best realised test E[Y]) ===")
    for k in ["IPW-X-X", "Direct-X-X", "IPW-O-X", "Hajek-O-X"]:
        if k in c5:
            b2, b5 = max(curves[k]), max(c5[k]); cmp[k] = (b2, b5)
            print("  %-12s N2000=%+.3f  N5000=%+.3f  Δ=%+.3f" % (k, b2, b5, b2 - b5))

def ser(idl, color, y): return {"id": idl, "label": idl, "color": color, "y": y}
COL = {"IPW-X-X": "#1f77b4", "IPW-O-X": "#1f77b4", "Hajek-O-X": "#9467bd",
       "DoublyRobust-X-X": "#2ca02c", "DoublyRobust-O-X": "#2ca02c", "Direct-X-X": "#ff7f0e", "Oracle": "#444444"}
HN = {"x": [float(g) for g in GAMMAS], "xmin": 1.0, "xmax": float(GAMMAS[-1]), "xlabel": "Γ  (sensitivity-model strength)",
      "ylabel": "realised E[Y] on test (N=5000)", "ymin": -0.06, "ymax": 0.10,
      "series": [ser(k, COL[k], curves[k]) for k in ["IPW-X-X", "Direct-X-X", "DoublyRobust-X-X", "IPW-O-X", "Hajek-O-X", "DoublyRobust-O-X", "Oracle"]],
      "hlines": [{"y": curves["Oracle"][0], "color": "#94a3b8", "dash": "4,4", "label": "oracle ceiling"}],
      "note": "Realised test E[Y] vs Γ, ALL methods trained at N=2000 (N_test=5000). Flat: IPW-X-X/Direct-X-X/DoublyRobust-X-X/Oracle. No Wasserstein here."}
(HERE / "hr_methods_n2000.json").write_text(json.dumps({"gammas": GAMMAS, "curves": curves, "cmp_vs_n5000": cmp}, indent=2))

# PNG
fig, ax = plt.subplots(figsize=(7.8, 4.6))
STY = {"IPW-X-X": ("#1f77b4", "--", ""), "Direct-X-X": ("#ff7f0e", "--", ""), "DoublyRobust-X-X": ("#2ca02c", "--", ""),
       "IPW-O-X": ("#1f77b4", "-", "s"), "Hajek-O-X": ("#9467bd", "-", "s"), "DoublyRobust-O-X": ("#2ca02c", "-", "s"), "Oracle": ("#444444", ":", "")}
for nm in ["IPW-X-X", "Direct-X-X", "DoublyRobust-X-X", "IPW-O-X", "Hajek-O-X", "DoublyRobust-O-X", "Oracle"]:
    c, ls, mk = STY[nm]; ax.plot(GAMMAS, curves[nm], color=c, ls=ls, marker=mk, ms=5, lw=2.2, label=nm)
ax.set_xlabel("Γ"); ax.set_ylabel("realised E[Y] on test (N=5000)"); ax.grid(alpha=0.25)
ax.legend(fontsize=7.5, ncol=2); ax.set_title("HR x_varying — realised test outcome vs Γ, all methods @ N_train=2000", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "methods_vs_gamma_n2000.png", dpi=130); plt.close(fig)

# splice
idx = ROOT / "index.html"; h = idx.read_text()
line = "var HR_N2000 = %s;" % json.dumps(HN, separators=(",", ":"))
if re.search(r'^var HR_N2000 = .*;$', h, flags=re.M):
    h = re.sub(r'^var HR_N2000 = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var HR_VALUE_GAMMA = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
call = "renderChart('ichart-hrfast-n2000',HR_N2000);"
if call not in h:
    h = h.replace("renderChart('ichart-hrfast-vgamma',HR_VALUE_GAMMA);",
                  "renderChart('ichart-hrfast-vgamma',HR_VALUE_GAMMA);\n" + call, 1)
idx.write_text(h)
print("\nspliced HR_N2000 + renderChart call into index.html")

# ---- rebuild the π(treat|X) policy chart (HR_POLICY) at N=2000 with all six methods ----
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
COLp = {"IPW-X-X": "#1f77b4", "Direct-X-X": "#ff7f0e", "DoublyRobust-X-X": "#2ca02c",
        "IPW-O-X": "#1f77b4", "Hajek-O-X": "#9467bd", "DoublyRobust-O-X": "#2ca02c", "Oracle": "#444444"}
def pser(idl, y): return {"id": idl, "label": ("Oracle (train iff X>0)" if idl == "Oracle" else idl),
                          "color": COLp[idl], "y": [round(float(v), 4) for v in y]}
def psa(g):
    return [pser("IPW-X-X", pol_ipw), pser("Direct-X-X", pol_dir), pser("DoublyRobust-X-X", pol_drx),
            pser("IPW-O-X", ox[g]), pser("Hajek-O-X", hox[g]), pser("DoublyRobust-O-X", dro[g]), pser("Oracle", oracle)]
HRP = {"x": [float(x) for x in GRID], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (seniority: −1 junior → +1 senior)",
       "ylabel": "π(treat | X)", "ymin": -0.05, "ymax": 1.05,
       "gammas": [float(g) for g in GAMMAS], "defaultGamma": 12.0, "matchedGamma": 12.0,
       "seriesByGamma": {gk(g): psa(g) for g in GAMMAS},
       "note": "π(treat|X) at N_train=2000, seed 0. Γ-free: IPW-X-X / Direct-X-X / DoublyRobust-X-X / Oracle; box-robust IPW-O-X / Hajek-O-X / DoublyRobust-O-X move with Γ."}
h = idx.read_text()
h = re.sub(r'^var HR_POLICY = .*;$', lambda m: "var HR_POLICY = %s;" % json.dumps(HRP, separators=(",", ":")), h, count=1, flags=re.M)
idx.write_text(h)
print("rebuilt HR_POLICY (π chart) at N=2000")

GA = np.asarray(GRID)
print("\n=== π treat-fraction breakdown @ N=2000 (overall / X<0 / X>=0) ===")
for nm, pg in [("IPW-X-X", pol_ipw), ("Direct-X-X", pol_dir), ("DoublyRobust-X-X", pol_drx),
               ("IPW-O-X@1", ox[1]), ("IPW-O-X@12", ox[12]), ("Hajek-O-X@1", hox[1]), ("Hajek-O-X@12", hox[12]),
               ("DoublyRobust-O-X@1", dro[1]), ("DoublyRobust-O-X@12", dro[12]), ("Oracle", oracle)]:
    y = np.asarray(pg); print("  %-20s %.2f / %.2f / %.2f" % (nm, y.mean(), y[GA < 0].mean(), y[GA >= 0].mean()))
