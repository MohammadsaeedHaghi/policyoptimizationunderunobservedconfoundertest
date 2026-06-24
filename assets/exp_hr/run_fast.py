"""Run the FAST baselines (IPW-X-X, Direct-X-X) on the HR x_varying data, N_train=2000, single seed.
Capture each method's optimal policy π(treat|X) over the 21-X grid + the oracle, build the interactive
var HR_POLICY (method-toggle dropdown, family colour code), splice into index.html, save JSON + PNG.
IPW-X-X uses the propensity estimated by COUNTING (common.count_ipw_weights_from_data) — exact on discrete X.
Usage: python3 run_fast.py"""
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

K, N, SEED = 2, 5000, 0
X = d.X_GRID
obs, full = d.generate_data(N, SEED, "x_varying")
w, P = common.count_ipw_weights_from_data(obs["X"], obs["T"], K)        # counting propensity -> Hájek IPW weights
xr = np.round(obs["X"], 2)
def grid_of(pi): return [round(float(pi[1, np.where(xr == round(float(gx), 2))[0][0]]), 4) for gx in X]

pol_ipw = grid_of(ipwxx(obs["X"], obs["T"], obs["Y"], w, n_arms=K, discretize=False).pi)
pol_dir = grid_of(directxx(obs["X"], obs["T"], obs["Y"], n_arms=K, discretize=False).pi)
oracle = [float(v) for v in d.oracle_policy(X)]
# IPW-O-X (box-robust value) swept over Γ — SAME counting-propensity weights w as IPW-X-X
GAMMAS = [1, 2, 4, 8, 12]
pol_ox = {g: grid_of(ipwox(obs["X"], obs["T"], obs["Y"], w, n_arms=K, Gamma=float(g), discretize=False).pi) for g in GAMMAS}
# Hajek-O-X (box-robust REGRET) swept over Γ — needs the RAW inverse weights 1/ê (≥1) from the SAME counting propensity
w_raw = 1.0 / common.factual_propensity(P, obs["T"])
pol_hox = {g: grid_of(hajekox(obs["X"], obs["T"], obs["Y"], w_raw, n_arms=K, Gamma=float(g), maximize=True, discretize=False).pi) for g in GAMMAS}

rv = {"IPW-X-X": d.realized_value(pol_ipw), "Direct-X-X": d.realized_value(pol_dir), "Oracle": d.realized_value(oracle)}
rv.update({"IPW-O-X@G%d" % g: d.realized_value(pol_ox[g]) for g in GAMMAS})
rv.update({"Hajek-O-X@G%d" % g: d.realized_value(pol_hox[g]) for g in GAMMAS})
print("realised E[Y]:  IPW-X-X=%.3f   Direct-X-X=%.3f   Oracle=%.3f" % (rv["IPW-X-X"], rv["Direct-X-X"], rv["Oracle"]))
print("IPW-O-X by Γ:   " + "  ".join("Γ%d=%.3f(treat%.2f)" % (g, rv["IPW-O-X@G%d" % g], float(np.mean(pol_ox[g]))) for g in GAMMAS))
print("Hajek-O-X by Γ: " + "  ".join("Γ%d=%.3f(treat%.2f)" % (g, rv["Hajek-O-X@G%d" % g], float(np.mean(pol_hox[g]))) for g in GAMMAS))
print("counting ê(X)=P(T=1|X): mean=%.3f range=[%.3f,%.3f]" % (P[:, 1].mean(), P[:, 1].min(), P[:, 1].max()))

def ser(idl, label, color, y):
    return {"id": idl, "label": label, "color": color, "y": [round(float(v), 4) for v in y]}
def gkey(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
# family colours; renderChart's _mstyle gives -X-X dashed, -O-X solid+squares, Oracle dotted
def series_at(g):
    return [ser("IPW-X-X", "IPW-X-X", "#1f77b4", pol_ipw),
            ser("Direct-X-X", "Direct-X-X", "#ff7f0e", pol_dir),
            ser("IPW-O-X", "IPW-O-X", "#1f77b4", pol_ox[g]),
            ser("Hajek-O-X", "Hajek-O-X", "#9467bd", pol_hox[g]),
            ser("Oracle", "Oracle (train iff X>0)", "#444444", oracle)]
HRP = {"x": [float(x) for x in X], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (seniority: −1 junior → +1 senior)",
       "ylabel": "π(treat | X)", "ymin": -0.05, "ymax": 1.05,
       "gammas": [float(g) for g in GAMMAS], "defaultGamma": 12.0, "matchedGamma": 12.0,
       "seriesByGamma": {gkey(g): series_at(g) for g in GAMMAS},
       "note": "IPW-X-X & Direct-X-X (fast, Γ-free) + box-robust IPW-O-X swept over Γ; only IPW-O-X moves with Γ. N=5000, seed 0."}
(HERE / "hr_fast_policy.json").write_text(json.dumps(HRP))
(HERE / "hr_fast_realised.json").write_text(json.dumps(rv, indent=2))

# PNG
fig, ax = plt.subplots(figsize=(7.6, 4.2))
ax.step(X, pol_ipw, where="mid", lw=2.4, color="#1f77b4", ls="--", label="IPW-X-X")
ax.step(X, pol_dir, where="mid", lw=2.4, color="#ff7f0e", ls="--", label="Direct-X-X")
ax.step(X, pol_ox[12], where="mid", lw=2.2, color="#1f77b4", ls="-", marker="s", ms=4, label="IPW-O-X (Γ=12)")
ax.step(X, pol_hox[12], where="mid", lw=2.2, color="#9467bd", ls="-", marker="s", ms=4, label="Hajek-O-X (Γ=12)")
ax.step(X, oracle, where="mid", lw=2.0, color="#444444", ls=":", label="Oracle (train iff X>0)")
ax.axvline(0, color="#cbd5e1", lw=1.0); ax.set_xlabel("X"); ax.set_ylabel("π(treat | X)"); ax.set_ylim(-0.1, 1.1)
ax.legend(fontsize=8); ax.grid(alpha=0.25); ax.set_title("HR x_varying — fast-method policies (N=5000, seed 0)", fontsize=9)
fig.tight_layout(); fig.savefig(HERE / "fast_policy.png", dpi=130); plt.close(fig)

# splice var + call (use a non 'ichart-hr-' id so build.py's HR-call strip won't remove it)
idx = ROOT / "index.html"; h = idx.read_text()
line = "var HR_POLICY = %s;" % json.dumps(HRP, separators=(",", ":"))
if re.search(r'^var HR_POLICY = .*;$', h, flags=re.M):
    h = re.sub(r'^var HR_POLICY = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var HR_CHARTS = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
call = "renderChart('ichart-hrfast-policy',HR_POLICY);"
if call not in h:
    h = h.replace("renderChart('ichart-hr-counts-x',HR_CHARTS.counts_x);",
                  "renderChart('ichart-hr-counts-x',HR_CHARTS.counts_x);\n" + call, 1)
idx.write_text(h)
print("spliced HR_POLICY + renderChart call into index.html")
