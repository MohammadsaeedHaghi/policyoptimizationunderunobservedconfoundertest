"""Build the exp_owwin charts (data-explanation + value-vs-Γ + π policy) and result table, splice var OWWIN_CHARTS
+ renderChart calls into index.html. Run AFTER run_owwin.py. Color=family, shape=uncertainty set."""
import sys, json, re, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin"
sp = importlib.util.spec_from_file_location("owdgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["owdgp"] = d; sp.loader.exec_module(d)
R = json.loads((HERE / "owwin_results.json").read_text())
G = np.array(R["grid"]); GAMMAS = R["gammas"]; t = d.grid_truth()
FAM = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Direct": "#ff7f0e", "Hajek": "#9467bd", "Oracle": "#444444"}
BLUE, RED, BLUE_DK, RED_DK = "#1f77b4", "#dc2626", "#16537e", "#9a1b1b"
def fam(m): return m.split("-")[0]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
def ser(idl, label, color, y, **kw):
    dd = {"id": idl, "label": label, "color": color, "y": [None if v is None else round(float(v), 4) for v in y]}; dd.update(kw); return dd
def chart(xlab, ylab, ymin, ymax, series, note, hlines=None, x=None):
    c = {"x": [float(v) for v in (x if x is not None else G)], "xmin": -1.0, "xmax": 1.0, "xlabel": xlab, "ylabel": ylab, "ymin": ymin, "ymax": ymax, "series": series, "note": note}
    if hlines: c["hlines"] = hlines
    return c
W = {}
W["outcome"] = chart("X  (discrete covariate, 7 levels)", "E[Y(t) | X, S]", -3.2, 3.7, [
    ser("m1p", "E[Y(1)|X,S=+1]", BLUE, t["m1p"], marker="square"), ser("m1m", "E[Y(1)|X,S=−1]", BLUE, t["m1m"], marker="circle"),
    ser("m1a", "E[Y(1)|X] avg", BLUE_DK, t["eY1"], width=4.4),
    ser("m0p", "E[Y(0)|X,S=+1]", RED, t["m0p"], marker="square"), ser("m0m", "E[Y(0)|X,S=−1]", RED, t["m0m"], marker="circle"),
    ser("m0a", "E[Y(0)|X] avg", RED_DK, t["eY0"], width=4.4),
], "Blue=treated, red=control. Square=S=+1, circle=S=−1, bold=avg over S. S shifts BOTH arms by 2.5 (strong outcome confounding); treatment adds X.")
W["cate"] = chart("X", "CATE(X)=E[Y(1)−Y(0)|X]", -1.1, 1.1, [ser("cate", "CATE(X)=X", "#4f46e5", t["cate"], width=3.2)],
    "CATE(X)=X for both S, so the optimal policy is treat iff X>0.", hlines=[{"y": 0.0, "color": "#94a3b8", "dash": "4,4", "label": "no effect"}])
W["sx"] = chart("X", "P(S=+1 | X)", 0.0, 1.0, [ser("ps1", "P(S=+1|X)=σ(6X)", "#7c3aed", t["p_s1"], width=3.2)],
    "THE lever: the unobserved S is strongly correlated with X, so balancing X also balances the hidden S.", hlines=[{"y": 0.5, "color": "#94a3b8", "dash": "4,4", "label": "50/50"}])
W["prop"] = chart("X", "P(treat | X, S)", 0.0, 1.0, [
    ser("ep", "e(X,S=+1)", BLUE, t["e_plus"], marker="square"), ser("em", "e(X,S=−1)", RED, t["e_minus"], marker="circle"),
    ser("emarg", "marginal ẽ(X)", "#334155", t["e_marg"], width=3.4)],
    "Mis-targeted (treats low X) + S-confounded. We FIT a logistic P(T|X) ⇒ mis-specified ⇒ tight ε>0 even on discrete X.")
# observed imbalance from one draw
obs, _ = d.generate(R["N_train"], 0); xr = obs["X"].ravel()
ftr = np.array([np.mean(xr[obs["T"] == 1] == c) for c in G]); fco = np.array([np.mean(xr[obs["T"] == 0] == c) for c in G])
W["obs"] = chart("X", "within-arm frequency (N=%d)" % R["N_train"], 0.0, float(max(ftr.max(), fco.max()) * 1.15), [
    ser("tr", "treated (T=1)", "#0e7490", ftr, marker="square"), ser("co", "control (T=0)", "#94a3b8", fco, marker="circle")],
    "Observed imbalance per level: treated skew LOW X, control HIGH X. The residual imbalance the Wasserstein term corrects.")

# value-vs-Γ (8-seed mean)
order = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
orc = R["oracle"]
W["value"] = {"x": [float(g) for g in GAMMAS], "xmin": float(min(GAMMAS)), "xmax": float(max(GAMMAS)),
    "xlabel": "Γ  (sensitivity-model strength)", "ylabel": "realised E[Y] (8-seed mean)", "ymin": 0.0, "ymax": round(orc + 0.03, 3),
    "series": [ser(m, m, FAM[fam(m)], R["mean"][m]) for m in order] + [ser("Oracle", "Oracle", FAM["Oracle"], [orc] * len(GAMMAS))],
    "hlines": [{"y": orc, "color": "#94a3b8", "dash": "4,4", "label": "oracle"}],
    "note": "CAPPED (treat<=50%%): realised test E[Y] vs Γ, N_train=%d, 8-seed mean. Colour=family, shape=set (XX dashed, OX squares, OW circles). Clean OW>OX>XX at every Γ." % R["N_train"]}
# policy (seed 0) with Γ dropdown
P = R["policy_seed0"]; oracle = d.oracle_policy(G)
def psa(g):
    k = gk(g)
    return [ser(m, m, FAM[fam(m)], P[m][k]) for m in order] + [ser("Oracle", "Oracle (treat iff X>0)", FAM["Oracle"], [round(float(v), 4) for v in oracle])]
W["policy"] = {"x": [float(v) for v in G], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (discrete covariate)", "ylabel": "π(treat | X)",
    "ymin": -0.05, "ymax": 1.05, "gammas": [float(g) for g in GAMMAS], "defaultGamma": 8.0, "matchedGamma": 8.0,
    "seriesByGamma": {gk(g): psa(g) for g in GAMMAS},
    "note": "Deployed π(treat|X) (seed 0), CAPPED. Γ dropdown. OW tracks treat-iff-X>0 (oracle); the box methods under-treat the good high-X region."}
(HERE / "owwin_charts.json").write_text(json.dumps(W))

# result table (mean ± SD), bold family winners
def fmt(v): return "%.3f" % v
def row(m):
    mu = R["mean"][m]; sdv = R["sd"][m]
    cells = "".join("<td>%s&plusmn;%s</td>" % (fmt(mu[i]), fmt(sdv[i])) for i in range(len(GAMMAS)))
    return "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (FAM[fam(m)], m, cells)
head = "<tr><th>method</th>" + "".join("<th>Γ=%s</th>" % gk(g) for g in GAMMAS) + "</tr>"
orow = "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>Oracle (best policy)</td>%s</tr>" % (FAM["Oracle"], "".join("<td>%s</td>" % fmt(orc) for _ in GAMMAS))
tbl = "<table class='restab'>%s%s%s</table>\n<p class='muted' style='margin:6px 0 0'>CAPPED (treat &le; 50%%): realised test E[Y] by Γ, N_train=%d, mean &plusmn; SD over 8 seeds. XX flat (Γ-free); OX and OW sweep Γ. Per-seed both families are 8/8 for OW&gt;OX&gt;XX.</p>" % (head, "".join([row(m) for m in order]), orow, R["N_train"])
(HERE / "owwin_table.html").write_text(tbl)

# PNGs for the chartable ones
def png(name, c):
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); MK = {"square": "s", "circle": "o"}
    for s in c["series"]:
        n = s["id"]; ls = "--" if n.endswith("-X-X") else (":" if n == "Oracle" else "-")
        ax.plot(c["x"], s["y"], color=s["color"], ls=ls, marker=MK.get(s.get("marker"), "s" if n.endswith("-O-X") else ("o" if n.endswith("-O-W") else "")), ms=4, lw=s.get("width", 2.0), label=s["label"])
    for hl in c.get("hlines", []): ax.axhline(hl["y"], ls="--", lw=1, color=hl["color"])
    ax.set_xlabel(c["xlabel"]); ax.set_ylabel(c["ylabel"]); ax.set_ylim(c["ymin"], c["ymax"]); ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.25); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / (name + ".png"), dpi=120); plt.close(fig)
for nm, c in W.items():
    if "seriesByGamma" not in c: png(nm, c)

# splice
idx = ROOT / "index.html"; h = idx.read_text()
line = "var OWWIN_CHARTS = %s;" % json.dumps(W, separators=(",", ":"))
if re.search(r'^var OWWIN_CHARTS = .*;$', h, flags=re.M):
    h = re.sub(r'^var OWWIN_CHARTS = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var PRIME_DATATELL_SUM = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
calls = "\n".join("renderChart('ichart-ow-%s',OWWIN_CHARTS.%s);" % (k, k) for k in W)
h = re.sub(r"\nrenderChart\('ichart-ow-[^\n]*", "", h)
if "renderChart('ichart-ow-outcome'" not in h:
    anchor = "renderChart('ichart-p10k-datatell-sum',PRIME_DATATELL_SUM);"
    assert anchor in h, "anchor missing"
    h = h.replace(anchor, anchor + "\n" + calls, 1)
# embed table
assert h.count("OWWIN_TABLE") == 1, "need exactly one OWWIN_TABLE placeholder (%d)" % h.count("OWWIN_TABLE")
h = h.replace("OWWIN_TABLE", tbl)
idx.write_text(h)
print("spliced OWWIN_CHARTS (%d) + table; mean OW(DR)=%s" % (len(W), R["mean"]["DoublyRobust-O-W"]))
