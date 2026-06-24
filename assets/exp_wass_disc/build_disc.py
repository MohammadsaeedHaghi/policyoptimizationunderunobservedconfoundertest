"""Build the DISC_CHARTS (discrete-X 'Wasserstein wins') and splice them + renderChart calls into index.html.
Ground-truth curves from dgp.grid_truth(); value/policy from disc_results.json (run_disc.py).  Saves PNGs too."""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_wass_disc"
sys.path.insert(0, str(ROOT)); import common
sp = importlib.util.spec_from_file_location("ddgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["ddgp"] = d; sp.loader.exec_module(d)

R = json.loads((HERE / "disc_results.json").read_text())
GAMMAS = R["gammas"]; G = np.array(R["grid"]); t = d.grid_truth()
m1p, m1m = d.mu1(G, 1.0), d.mu1(G, -1.0)
m0p, m0m = d.mu0(G, 1.0), d.mu0(G, -1.0)
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
def ser(idl, label, color, y, **kw):
    dd = {"id": idl, "label": label, "color": color, "y": [round(float(v), 4) for v in y]}; dd.update(kw); return dd
def chart(xlab, ylab, ymin, ymax, series, note, hlines=None, x=None):
    c = {"x": [float(v) for v in (x if x is not None else G)], "xmin": -1.0, "xmax": 1.0, "xlabel": xlab, "ylabel": ylab,
         "ymin": ymin, "ymax": ymax, "series": series, "note": note}
    if hlines: c["hlines"] = hlines
    return c
BLUE, RED, BLUE_DK, RED_DK = "#1f77b4", "#dc2626", "#16537e", "#9a1b1b"
W = {}
W["outcome"] = chart("X  (discrete covariate, 9 levels)", "E[Y(t) | X, S]", -2.2, 2.2, [
    ser("m1p", "E[Y(1)|X,S=+1]  treated", BLUE, m1p, marker="square"),
    ser("m1m", "E[Y(1)|X,S=−1]  treated", BLUE, m1m, marker="circle"),
    ser("m1a", "E[Y(1)|X]  treated · avg", BLUE_DK, t["eY1"], width=4.6),
    ser("m0p", "E[Y(0)|X,S=+1]  control", RED, m0p, marker="square"),
    ser("m0m", "E[Y(0)|X,S=−1]  control", RED, m0m, marker="circle"),
    ser("m0a", "E[Y(0)|X]  control · avg", RED_DK, t["eY0"], width=4.6),
], "Blue=treated, red=control. Square=S=+1, circle=S=−1, bold=average over S. Identical recipe to the continuous DGP, only X is now on 9 discrete levels.")
W["cate"] = chart("X", "CATE(X) = E[Y(1)−Y(0) | X]", -1.1, 1.1, [
    ser("cate", "CATE(X) = X", "#4f46e5", t["cate"], width=3.4),
], "CATE(X)=X (same for both S), so the optimal policy is train iff X>0.",
   hlines=[{"y": 0.0, "color": "#94a3b8", "dash": "4,4", "label": "no effect"}])
W["sx"] = chart("X", "P(S=+1 | X)", 0.0, 1.0, [
    ser("ps1", "P(S=+1 | X) = σ(4X)", "#7c3aed", t["p_s1"], width=3.4),
], "The KEY: the unobserved S is strongly correlated with X, so balancing X also balances S.",
   hlines=[{"y": 0.5, "color": "#94a3b8", "dash": "4,4", "label": "50/50"}])
W["prop"] = chart("X", "P(enroll | X, S)", 0.0, 1.0, [
    ser("ep", "e(X,S=+1)", BLUE, t["e_plus"], marker="square"),
    ser("em", "e(X,S=−1)", RED, t["e_minus"], marker="circle"),
    ser("emarg", "marginal ẽ(X)", "#334155", t["e_marg"], width=3.6),
], "Mis-targeted (treats low X) + S-confounded. We FIT a logistic P(T|X) ⇒ mis-specified ⇒ the tight ε>0 even though X is discrete.")

# observed-data diagnostic on the SAME training draw (generate(1000,0))
obs, _ = d.generate(R["N_train"], 0); xr = obs["X"].ravel()
lv = d.LEVELS
htr = np.array([np.mean((xr[obs["T"] == 1] == c)) for c in lv])
hco = np.array([np.mean((xr[obs["T"] == 0] == c)) for c in lv])
W["obs"] = chart("X", "within-arm frequency  (N=%d)" % R["N_train"], 0.0, float(max(htr.max(), hco.max()) * 1.15), [
    ser("tr", "treated (T=1)", "#0e7490", htr, marker="square"), ser("co", "control (T=0)", "#94a3b8", hco, marker="circle"),
], "Observed imbalance per level: treated skew to LOW X, control to HIGH X. Logistic IPW cannot fully erase it ⇒ ε>0.", x=lv)

# value-vs-Γ (W-vs-box comparison)
COL = {"IPW-X-X": "#1f77b4", "IPW-O-X": "#1f77b4", "DoublyRobust-X-X": "#2ca02c",
       "DoublyRobust-O-X": "#2ca02c", "IPW-O-W": "#1f77b4", "DoublyRobust-O-W": "#2ca02c", "Oracle": "#444444"}
V = R["value_by_gamma"]; orc = R["oracle"]
W["value"] = {"x": [float(g) for g in GAMMAS], "xmin": 1.0, "xmax": 8.0, "xlabel": "Γ  (sensitivity-model strength)",
              "ylabel": "realised E[Y] on test", "ymin": 0.10, "ymax": 0.30,
              "series": [ser("IPW-X-X", "IPW-X-X", COL["IPW-X-X"], V["IPW-X-X"]),
                         ser("DoublyRobust-X-X", "DoublyRobust-X-X", COL["DoublyRobust-X-X"], V["DoublyRobust-X-X"]),
                         ser("IPW-O-X", "IPW-O-X", COL["IPW-O-X"], V["IPW-O-X"]),
                         ser("DoublyRobust-O-X", "DoublyRobust-O-X", COL["DoublyRobust-O-X"], V["DoublyRobust-O-X"]),
                         ser("IPW-O-W", "IPW-O-W", COL["IPW-O-W"], V["IPW-O-W"]),
                         ser("DoublyRobust-O-W", "DoublyRobust-O-W", COL["DoublyRobust-O-W"], V["DoublyRobust-O-W"]),
                         ser("Oracle", "Oracle", COL["Oracle"], [orc] * len(GAMMAS))],
              "hlines": [{"y": orc, "color": "#94a3b8", "dash": "4,4", "label": "oracle"}],
              "note": "Realised test E[Y] vs Γ, discrete X, N_train=%d, logistic propensity (ε>0). DoublyRobust-O-W (green circles) dominates DoublyRobust-O-X (green squares) at every Γ; the box-only DR collapses as Γ grows." % R["N_train"]}
# policy chart with Γ dropdown
P = R["policy_by_gamma"]; oracle = d.oracle_policy(G)
def psa(g):
    k = gk(g)
    return [ser("IPW-O-X", "IPW-O-X", COL["IPW-O-X"], P["IPW-O-X"][k]),
            ser("DoublyRobust-O-X", "DoublyRobust-O-X", COL["DoublyRobust-O-X"], P["DoublyRobust-O-X"][k]),
            ser("IPW-O-W", "IPW-O-W", COL["IPW-O-W"], P["IPW-O-W"][k]),
            ser("DoublyRobust-O-W", "DoublyRobust-O-W", COL["DoublyRobust-O-W"], P["DoublyRobust-O-W"][k]),
            ser("Oracle", "Oracle (train iff X>0)", COL["Oracle"], [round(float(v), 4) for v in oracle])]
W["policy"] = {"x": [float(v) for v in G], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (discrete covariate)",
               "ylabel": "π(treat | X)", "ymin": -0.05, "ymax": 1.05, "gammas": [float(g) for g in GAMMAS],
               "defaultGamma": 8.0, "matchedGamma": 8.0, "seriesByGamma": {gk(g): psa(g) for g in GAMMAS},
               "note": "Deployed π(treat|X), discrete X, N_train=%d. Γ dropdown; at high Γ the box-only DR-O-X over-hedges, Wasserstein keeps a sane treat-iff-X>0 shape." % R["N_train"]}

(HERE / "disc_charts.json").write_text(json.dumps(W))

# PNGs
def png(name, c):
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); MK = {"square": "s", "circle": "o"}
    for s in c["series"]:
        ax.plot(c["x"], s["y"], marker=MK.get(s.get("marker"), ""), ms=4, lw=s.get("width", 2.0), color=s["color"], label=s["label"])
    for h in c.get("hlines", []): ax.axhline(h["y"], ls="--", lw=1, color=h["color"])
    ax.set_xlabel(c["xlabel"]); ax.set_ylabel(c["ylabel"]); ax.set_ylim(c["ymin"], c["ymax"]); ax.legend(fontsize=7); ax.grid(alpha=.25); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / (name + ".png"), dpi=120); plt.close(fig)
for nm, c in W.items():
    if "seriesByGamma" not in c: png(nm, c)

# splice DISC_CHARTS + renderChart calls
idx = ROOT / "index.html"; h = idx.read_text()
line = "var DISC_CHARTS = %s;" % json.dumps(W, separators=(",", ":"))
if re.search(r'^var DISC_CHARTS = .*;$', h, flags=re.M):
    h = re.sub(r'^var DISC_CHARTS = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var WASS_CHARTS = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
calls = "\n".join("renderChart('ichart-disc-%s',DISC_CHARTS.%s);" % (k, k) for k in W)
h = re.sub(r"\nrenderChart\('ichart-disc-[^\n]*", "", h)
if "renderChart('ichart-disc-outcome'" not in h:
    anchor = "renderChart('ichart-wass-policy',WASS_CHARTS.policy);"
    assert anchor in h, "wass-policy render call anchor not found"
    h = h.replace(anchor, anchor + "\n" + calls, 1)
idx.write_text(h)
print("spliced DISC_CHARTS + %d calls; ε_log=%s ε_cnt=%s" % (len(W), R["epsilon_logistic"], R["epsilon_counting"]))
