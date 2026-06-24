"""Build the HR-training DGP explainer: ground-truth + observed-data diagnostics at N_train=2000,
save artifacts (NPZ), render PNGs, build interactive renderChart vars (var HR_CHARTS) and splice into
index.html. NO method is run — this only explains the DGP and the observed data. Usage: python3 build.py"""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_hr"
sp = importlib.util.spec_from_file_location("hrdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sp.loader.exec_module(d)
X = d.X_GRID; N = 2000; SEED = 0
DGPS = ["x_varying"]            # only the seniority-adaptive variant is kept

# palette (these are NOT methods, so renderChart's _mstyle falls back to these explicit colours)
C = dict(splus="#2563eb", sminus="#dc2626", y1="#0e7490", y0="#b45309",
         marg="#4f46e5", obsT1="#0e7490", obsT0="#b45309", trueT1="#10b981", trueT0="#f59e0b",
         eplus="#2563eb", eminus="#dc2626", neutral="#64748b", gam="#7c3aed", naive="#dc2626", cate="#4f46e5",
         treated="#0e7490", control="#94a3b8")


def ser(idl, label, color, y, **kw):
    d = {"id": idl, "label": label, "color": color, "y": [round(float(v), 4) for v in y]}
    d.update(kw)                       # optional per-series style: marker ('square'|'circle'), width
    return d


def chart(xlab, ylab, ymin, ymax, series, note, hlines=None):
    c = {"x": [float(x) for x in X], "xmin": -1.0, "xmax": 1.0, "xlabel": xlab, "ylabel": ylab,
         "ymin": ymin, "ymax": ymax, "series": series, "note": note}
    if hlines: c["hlines"] = hlines
    return c


truth = {g: d.grid_truth(g) for g in DGPS}
HR = {}

# ---------- SHARED: outcome model + CATE (DGP-independent) ----------
t = truth["x_varying"]
BLUE, RED = "#1f77b4", "#dc2626"        # blue = trained (T=1), red = untrained (T=0)
BLUE_DK, RED_DK = "#16537e", "#9a1b1b"  # darker shade for the bold average-over-S line
avg1 = 0.5 * (np.asarray(t["p_y1_splus"]) + np.asarray(t["p_y1_sminus"]))   # P(Y(1)=+1 | X)
avg0 = 0.5 * (np.asarray(t["p_y0_splus"]) + np.asarray(t["p_y0_sminus"]))   # P(Y(0)=+1 | X)
HR["outcome"] = chart("X  (seniority: −1 junior → +1 senior)", "P(promotion = +1)", 0.0, 1.0, [
    ser("y1sp", "P(Y(1)=+1 | X, S=+1)  trained · degree", BLUE, t["p_y1_splus"], marker="square"),
    ser("y1sm", "P(Y(1)=+1 | X, S=−1)  trained · no-degree", BLUE, t["p_y1_sminus"], marker="circle"),
    ser("y1av", "P(Y(1)=+1 | X)  trained · average over S", BLUE_DK, avg1, width=4.8),
    ser("y0sp", "P(Y(0)=+1 | X, S=+1)  untrained · degree", RED, t["p_y0_splus"], marker="square"),
    ser("y0sm", "P(Y(0)=+1 | X, S=−1)  untrained · no-degree", RED, t["p_y0_sminus"], marker="circle"),
    ser("y0av", "P(Y(0)=+1 | X)  untrained · average over S", RED_DK, avg0, width=4.8),
], "Blue = trained (T=1), red = untrained (T=0). Square = degree (S=+1), circle = no-degree (S=−1), bold line = average over S.")

HR["cate"] = chart("X", "CATE = E[Y(1) − Y(0) | X]   (Y ∈ {−1,+1})", -1.2, 1.2, [
    ser("cate", "marginal CATE (over S)", C["marg"], t["cate"]),
    ser("catep", "CATE | S=+1 (degree)", C["splus"], t["cate_splus"]),
    ser("catem", "CATE | S=−1 (no-degree)", C["sminus"], t["cate_sminus"]),
], "Marginal CATE crosses 0 at X=0 → TRUE policy is train iff X>0.", hlines=[{"y": 0.0, "color": "#94a3b8", "dash": "4,4", "label": "no effect"}])

# ---------- PER-DGP ----------
diag = {}
for g in DGPS:
    t = truth[g]
    sfx = "u" if g == "uniform" else "x"
    obs, full = d.generate_data(N, SEED, g)
    xr = np.round(obs["X"], 2)
    nt = np.array([int(np.sum((xr == round(float(x), 2)) & (obs["T"] == 1))) for x in X])
    nc = np.array([int(np.sum((xr == round(float(x), 2)) & (obs["T"] == 0))) for x in X])
    fooled = int(np.sum((X != 0) & ((t["naive"] > 0) != (t["oracle"] > 0))))
    diag[g] = dict(
        oracle=d.realized_value(t["oracle"]), alltreat=d.realized_value(np.ones(21)),
        allctrl=d.realized_value(np.zeros(21)), naive=d.realized_value((t["naive"] > 0).astype(float)),
        fooled=fooled, n_treated=int(obs["T"].sum()), n_control=int((1 - obs["T"]).sum()),
        gamma=[round(float(v), 2) for v in t["gamma"][[0, 5, 10, 15, 20]]])

    HR["prop_" + sfx] = chart("X", "P(enroll in training | X, S)", 0.0, 1.0, [
        ser("ep", "e(X, S=+1)  degree-holder", C["eplus"], t["eplus"]),
        ser("em", "e(X, S=−1)  no-degree", C["eminus"], t["eminus"]),
    ], "Marginal ẽ(X)=0.5 everywhere (dashed), so confounding is invisible to a model of P(T|X).",
       hlines=[{"y": 0.5, "color": C["neutral"], "dash": "5,4", "label": "marginal ẽ(X)=0.5"}])

    HR["obs_" + sfx] = chart("X", "E[Y | X, T]   (promotion score, ±1)", -1.0, 1.0, [
        ser("oT1", "OBSERVED  E[Y | X, T=1]  (trained, confounded)", C["obsT1"], t["obsY_T1"]),
        ser("oT0", "OBSERVED  E[Y | X, T=0]  (untrained, confounded)", C["obsT0"], t["obsY_T0"]),
        ser("tT1", "TRUTH  E[Y(1) | X]", C["trueT1"], t["eY1"]),
        ser("tT0", "TRUTH  E[Y(0) | X]", C["trueT0"], t["eY0"]),
    ], "What HR sees (solid) vs the truth (green/amber): selection on S bends the observed curves.")

    HR["naive_" + sfx] = chart("X", "effect estimate", -1.2, 1.2, [
        ser("naive", "NAIVE observed effect  E[Y|T=1]−E[Y|T=0]", C["naive"], t["naive"]),
        ser("cate", "TRUE marginal CATE", C["cate"], t["cate"]),
    ], "Naive (red) has the WRONG SIGN vs true CATE (indigo) at %d/21 atoms → Direct/IPW are fooled." % fooled,
       hlines=[{"y": 0.0, "color": "#94a3b8", "dash": "4,4", "label": "0"}])

    HR["select_" + sfx] = chart("X", "P(S=+1 | X, T)   degree-holder share", 0.0, 1.0, [
        ser("s1", "among TRAINED  (T=1)", C["treated"], t["pS1_T1"]),
        ser("s0", "among UNTRAINED (T=0)", C["control"], t["pS1_T0"]),
    ], "Why it's confounded: trained units are degree-heavy, untrained are degree-light.",
       hlines=[{"y": 0.5, "color": "#94a3b8", "dash": "4,4", "label": "50/50"}])

    HR["counts_" + sfx] = chart("X", "sample count  (N_train=2000, seed 0)", 0.0, float(max(nt.max(), nc.max()) + 5), [
        ser("nt", "treated (T=1)", C["treated"], nt),
        ser("nc", "control (T=0)", C["control"], nc),
    ], "Units observed per seniority level (≈%d treated / %d control overall)." % (int(nt.sum()), int(nc.sum())))

    # ---- artifacts ----
    np.savez(HERE / ("hr_%s_N%d.npz" % (g, N)), **{k: np.asarray(v) for k, v in t.items()},
             sample_X=obs["X"], sample_T=obs["T"], sample_Y=obs["Y"], sample_S=full["S"],
             n_treated=nt, n_control=nc)

(HERE / "hr_diagnostics.json").write_text(json.dumps(diag, indent=2))
(HERE / "hr_charts.json").write_text(json.dumps(HR))
print("diagnostics:", json.dumps(diag, indent=2))

# ---------- PNGs (static artifacts) ----------
_MK = {"square": "s", "circle": "o", "triangle": "^"}
def png(name, datac, kind="line"):
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for s in datac["series"]:
        ax.plot(datac["x"], s["y"], marker=_MK.get(s.get("marker"), ""), ms=4.2,
                lw=s.get("width", 2.0), color=s["color"], label=s["label"])
    for h in datac.get("hlines", []):
        ax.axhline(h["y"], ls="--", lw=1.1, color=h["color"])
    if any(s["y"] and (min(s["y"]) < 0) for s in datac["series"]):
        ax.axvline(0.0, color="#cbd5e1", lw=1.0, ls=":")
    ax.set_xlabel(datac["xlabel"]); ax.set_ylabel(datac["ylabel"]); ax.set_ylim(datac["ymin"], datac["ymax"])
    ax.legend(fontsize=7.2, framealpha=0.9); ax.grid(alpha=0.25); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / (name + ".png"), dpi=130); plt.close(fig)

for nm, c in HR.items():
    png(nm, c)
print("wrote %d PNGs + %d NPZ + chart JSON" % (len(HR), len(DGPS)))

# ---------- splice var HR_CHARTS into index.html ----------
idx = ROOT / "index.html"; h = idx.read_text()
line = "var HR_CHARTS = %s;" % json.dumps(HR, separators=(",", ":"))
if re.search(r'^var HR_CHARTS = .*;$', h, flags=re.M):
    h = re.sub(r'^var HR_CHARTS = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var POLICY_FIRST_N2000_S3 = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
# renderChart calls for every HR chart — strip any prior HR calls, then re-add the current set
h = re.sub(r"\nrenderChart\('ichart-hr-[^\n]*", "", h)
calls = "\n".join("renderChart('ichart-hr-%s',HR_CHARTS.%s);" % (k.replace("_", "-"), k) for k in HR)
h = h.replace("renderChart('ichart-first-n2000',POLICY_FIRST_N2000_S3);",
              "renderChart('ichart-first-n2000',POLICY_FIRST_N2000_S3);\n" + calls, 1)
idx.write_text(h)
print("spliced HR_CHARTS + %d renderChart calls into index.html" % len(HR))
