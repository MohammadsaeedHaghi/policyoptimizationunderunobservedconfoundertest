"""Build the data-generation explanation charts for the 'first prime' DGP (piecewise-linear first experiment),
splice PRIME_CHARTS + renderChart calls into index.html, and save PNGs.  No methods are run here, this is the
data-explanation tab only."""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_first_prime"
sp = importlib.util.spec_from_file_location("pdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["pdgp"] = d; sp.loader.exec_module(d)

t = d.grid_truth(); G = t["X"]
BLUE, RED, BLUE_DK, RED_DK = "#1f77b4", "#dc2626", "#16537e", "#9a1b1b"
def ser(idl, label, color, y, **kw):
    dd = {"id": idl, "label": label, "color": color, "y": [round(float(v), 4) for v in y]}; dd.update(kw); return dd
def chart(xlab, ylab, ymin, ymax, series, note, hlines=None, x=None):
    c = {"x": [float(v) for v in (x if x is not None else G)], "xmin": -1.0, "xmax": 1.0, "xlabel": xlab, "ylabel": ylab,
         "ymin": ymin, "ymax": ymax, "series": series, "note": note}
    if hlines: c["hlines"] = hlines
    return c

W = {}
W["outcome0"] = chart("X  (employability score, 21 levels)", "P(Y(0)=1 | X, S)  control", -0.05, 1.08, [
    ser("y0s1", "S=1", RED, t["y0_s1"], marker="square"),
    ser("y0s0", "S=0", RED, t["y0_s0"], marker="circle"),
    ser("y0avg", "average over S", RED_DK, t["y0_avg"], width=4.6),
], "Control outcome P(Y(0)=1|X,S) = min{0.5 + 0.231 X + 0.5 S, 1}. The unobserved S lifts the baseline by 0.5 (square vs circle); bold = average over S.")
W["outcome1"] = chart("X  (employability score, 21 levels)", "P(Y(1)=1 | X, S)  treated", -0.05, 1.08, [
    ser("y1s1", "S=1", BLUE, t["y1_s1"], marker="square"),
    ser("y1s0", "S=0", BLUE, t["y1_s0"], marker="circle"),
    ser("y1avg", "average over S", BLUE_DK, t["y1_avg"], width=4.6),
], "Treated outcome P(Y(1)=1|X,S) = min{1[X>=0] + S, 1}: a step up at X=0, plus S. For S=1 it is 1 everywhere; for S=0 it is the step 1[X>=0]. Bold = average over S.")
W["cate"] = chart("X", "CATE(X) = mu_1(X) - mu_0(X)", -0.32, 0.32, [
    ser("cate", "CATE(X) (averaged over S)", "#4f46e5", t["cate"], width=3.4),
], "NON-MONOTONE effect: training HURTS for X<0 (CATE<0) and HELPS for X>=0 (CATE>0). The optimal rule is train iff X>=0.",
   hlines=[{"y": 0.0, "color": "#94a3b8", "dash": "4,4", "label": "no effect"}])
W["prop"] = chart("X", "pi^1(X, S) = P(T=1 | X, S)", 0.0, 0.82, [
    ser("es1", "S=1 : 0.500 + 0.231 X", BLUE, t["e_s1"], marker="square"),
    ser("es0", "S=0 : 0.158 + 0.111 X", RED, t["e_s0"], marker="circle"),
    ser("eavg", "marginal e(X) (avg over S)", "#334155", t["e_avg"], width=3.6),
], "Piecewise-linear propensity, one line per S. High-S units are far more likely to be treated (and they also have better outcomes): this is the unobserved confounding.")
W["policy"] = chart("X", "optimal action", -0.08, 1.08, [
    ser("opt", "Oracle: train iff X>=0", "#444444", t["oracle"], width=3.6),
], "The Bayes-optimal policy under full information: train exactly the units with X>=0 (where CATE>0).",
   hlines=[{"y": 0.5, "color": "#cbd5e1", "dash": "3,3", "label": ""}])

# observed-data diagnostic: within-arm frequency per X level (one training draw)
rng = np.random.default_rng(0); tr = d.generate(4000, rng); xr = tr.X.ravel()
ftr = np.array([np.mean(xr[tr.T == 1] == c) for c in G]); fco = np.array([np.mean(xr[tr.T == 0] == c) for c in G])
W["obs"] = chart("X", "within-arm frequency  (N=4000)", 0.0, float(max(ftr.max(), fco.max()) * 1.15), [
    ser("tr", "treated (T=1)", "#0e7490", ftr, marker="square"),
    ser("co", "control (T=0)", "#94a3b8", fco, marker="circle"),
], "Observed assignment: treated units skew to HIGH X (the propensity rises with X and S), so the arms have different X-distributions.")

(HERE / "prime_charts.json").write_text(json.dumps(W))

def png(name, c):
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); MK = {"square": "s", "circle": "o"}
    for s in c["series"]:
        ax.plot(c["x"], s["y"], marker=MK.get(s.get("marker"), ""), ms=4, lw=s.get("width", 2.0), color=s["color"], label=s["label"])
    for hl in c.get("hlines", []):
        if hl.get("label"): ax.axhline(hl["y"], ls="--", lw=1, color=hl["color"], label=hl["label"])
        else: ax.axhline(hl["y"], ls="--", lw=1, color=hl["color"])
    ax.set_xlabel(c["xlabel"]); ax.set_ylabel(c["ylabel"]); ax.set_ylim(c["ymin"], c["ymax"]); ax.legend(fontsize=7); ax.grid(alpha=.25); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / (name + ".png"), dpi=120); plt.close(fig)
for nm, c in W.items(): png(nm, c)

# splice PRIME_CHARTS + render calls
idx = ROOT / "index.html"; h = idx.read_text()
line = "var PRIME_CHARTS = %s;" % json.dumps(W, separators=(",", ":"))
if re.search(r'^var PRIME_CHARTS = .*;$', h, flags=re.M):
    h = re.sub(r'^var PRIME_CHARTS = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var DISC_CHARTS = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
calls = "\n".join("renderChart('ichart-prime-%s',PRIME_CHARTS.%s);" % (k, k) for k in W)
h = re.sub(r"\nrenderChart\('ichart-prime-[^\n]*", "", h)
if "renderChart('ichart-prime-outcome0'" not in h:
    anchor = "renderChart('ichart-disc-policy',DISC_CHARTS.policy);"
    assert anchor in h, "disc-policy render anchor missing"
    h = h.replace(anchor, anchor + "\n" + calls, 1)
idx.write_text(h)
print("spliced PRIME_CHARTS + %d render calls; charts: %s" % (len(W), list(W)))
