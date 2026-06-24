"""Build the exp_owwin3 (E3) tab: data-explanation charts + per-regime value-vs-Γ (x=log10 Γ, to 1000, shows the
collapse) + π policy + mean±SD tables. Splice OWWIN3_CHARTS + render calls + tables into index.html."""
import sys, json, re, math, importlib.util
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
HERE = ROOT / "assets" / "exp_owwin3"
sp = importlib.util.spec_from_file_location("ow3dgp", str(HERE / "dgp.py")); d = importlib.util.module_from_spec(sp); sys.modules["ow3dgp"] = d; sp.loader.exec_module(d)
R = json.loads((HERE / "owwin3_results.json").read_text())
G = np.array(R["grid"]); GAMMAS = R["gammas"]; t = d.grid_truth(); orc = R["oracle"]
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
W["outcome"] = chart("X  (discrete, 7 levels)", "E[Y(t) | X, S]", -9.0, 9.0, [
    ser("m1p", "E[Y(1)|X,S=+1]", BLUE, t["m1p"], marker="square"), ser("m1m", "E[Y(1)|X,S=−1]", BLUE, t["m1m"], marker="circle"),
    ser("m1a", "E[Y(1)|X] avg", BLUE_DK, t["eY1"], width=4.4),
    ser("m0p", "E[Y(0)|X,S=+1]", RED, t["m0p"], marker="square"), ser("m0m", "E[Y(0)|X,S=−1]", RED, t["m0m"], marker="circle"),
    ser("m0a", "E[Y(0)|X] avg", RED_DK, t["eY0"], width=4.4)],
    "Blue=treated, red=control. Square=S=+1, circle=S=−1, bold=avg over S. S shifts BOTH arms by 5 (strong confounding); treatment adds 3X.")
W["cate"] = chart("X", "CATE(X)=3X", -3.3, 3.3, [ser("cate", "CATE(X)=3X", "#4f46e5", t["cate"], width=3.2)],
    "CATE=3X for both S, so the optimal policy is treat iff X>0.", hlines=[{"y": 0.0, "color": "#94a3b8", "dash": "4,4", "label": "no effect"}])
W["sx"] = chart("X", "P(S=+1 | X)", 0.0, 1.0, [ser("ps1", "P(S=+1|X)=σ(7X)", "#7c3aed", t["p_s1"], width=3.2)],
    "Strong S-X correlation so balancing X reaches the hidden S.", hlines=[{"y": 0.5, "color": "#94a3b8", "dash": "4,4", "label": "50/50"}])
W["prop"] = chart("X", "P(treat | X, S)", 0.0, 1.0, [
    ser("ep", "e(X,S=+1)", BLUE, t["e_plus"], marker="square"), ser("em", "e(X,S=−1)", RED, t["e_minus"], marker="circle"),
    ser("emarg", "marginal ẽ(X)", "#334155", t["e_marg"], width=3.4)],
    "STRONG selection on S (e=σ(4S−2X)): a big confounding bias that fools naive XX into over-treating, even with no cap.")
obs, _ = d.generate(R["N_train"], 0); xr = obs["X"].ravel()
ftr = np.array([np.mean(xr[obs["T"] == 1] == c) for c in G]); fco = np.array([np.mean(xr[obs["T"] == 0] == c) for c in G])
W["obs"] = chart("X", "within-arm frequency (N=%d)" % R["N_train"], 0.0, float(max(ftr.max(), fco.max()) * 1.15), [
    ser("tr", "treated (T=1)", "#0e7490", ftr, marker="square"), ser("co", "control (T=0)", "#94a3b8", fco, marker="circle")],
    "Observed imbalance per level (the residual the Wasserstein term corrects).")

order = ["IPW-X-X", "DoublyRobust-X-X", "Direct-X-X", "IPW-O-X", "DoublyRobust-O-X", "Hajek-O-X", "IPW-O-W", "DoublyRobust-O-W"]
XL = [float(g) for g in GAMMAS]
def value_chart(reg):
    M = R["regimes"][reg]["mean"]
    return {"x": XL, "xmin": min(XL), "xmax": max(XL), "xlabel": "Γ  (sensitivity-model strength)", "ylabel": "realised E[Y] (5-seed mean)",
            "ymin": -0.03, "ymax": round(orc + 0.04, 3),
            "series": [ser(m, m, FAM[fam(m)], M[m], marker=("circle" if m.endswith("O-W") else ("square" if m.endswith("O-X") else None))) for m in order]
                       + [ser("Oracle", "Oracle", FAM["Oracle"], [orc] * len(GAMMAS))],
            "hlines": [{"y": orc, "color": "#94a3b8", "dash": "4,4", "label": "oracle"}, {"y": 0.0, "color": "#cbd5e1", "dash": "2,3", "label": "never-treat"}],
            "note": "%s: realised E[Y] vs Γ, N=%d, 5-seed mean. Both OW methods (circles) beat EVERY other method (incl plain DoublyRobust-X-X) at small Γ; matched Γ≈5 so the win is at Γ≈1.5-2." % ("UNCAPPED" if reg == "uncap" else "CAPPED (treat<=50%)", R["N_train"])}
def policy_chart(reg):
    PB = R["regimes"][reg]["policy_by_seed"]; oracle = d.oracle_policy(G)
    seedkeys = [str(s) for s in R["seeds"]] + ["avg"]
    def psa(seed, g):
        k = gk(g)
        return [ser(m, m, FAM[fam(m)], PB[m][seed][k], marker=("circle" if m.endswith("O-W") else ("square" if m.endswith("O-X") else None))) for m in order] + [ser("Oracle", "Oracle (treat iff X>0)", FAM["Oracle"], [round(float(v), 4) for v in oracle])]
    return {"x": [float(v) for v in G], "xmin": -1.0, "xmax": 1.0, "xlabel": "X  (discrete)", "ylabel": "π(treat | X)",
            "ymin": -0.05, "ymax": 1.05, "gammas": [float(g) for g in GAMMAS], "defaultGamma": 8.0, "matchedGamma": 8.0,
            "seeds": [{"key": s, "label": ("Average (5 seeds)" if s == "avg" else "Seed " + s)} for s in seedkeys], "defaultSeed": "avg",
            "seriesBySeedGamma": {s: {gk(g): psa(s, g) for g in GAMMAS} for s in seedkeys},
            "note": "%s deployed π(treat|X). Γ + seed dropdowns. Default is the across-seed average; pick a single seed to see its exact policy (note the policy chart and the value chart's 5-seed mean are different aggregations)." % ("UNCAPPED" if reg == "uncap" else "CAPPED")}
for reg in ("uncap", "cap"):
    W[reg + "_value"] = value_chart(reg); W[reg + "_policy"] = policy_chart(reg)
(HERE / "owwin3_charts.json").write_text(json.dumps(W))

def fmt(v): return "%.3f" % v
def table(reg):
    M = R["regimes"][reg]["mean"]; SD = R["regimes"][reg]["sd"]
    head = "<tr><th>method</th>" + "".join("<th>Γ=%s</th>" % gk(g) for g in GAMMAS) + "</tr>"
    rows = []
    for m in order:
        cells = "".join("<td>%s&plusmn;%s</td>" % (fmt(M[m][i]), fmt(SD[m][i])) for i in range(len(GAMMAS)))
        rows.append("<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (FAM[fam(m)], m, cells))
    orow = "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>Oracle</td>%s</tr>" % (FAM["Oracle"], "".join("<td>%s</td>" % fmt(orc) for _ in GAMMAS))
    cap = "treat &le; 100%" if reg == "uncap" else "treat &le; 50%"
    return "<table class='restab'>%s%s%s</table>\n<p class='muted' style='margin:6px 0 0'>%s: realised E[Y] by Γ, N_train=%d, mean &plusmn; SD over 5 seeds. XX flat (Γ-free); OX/OW sweep Γ. Past Γ≈8 the robust methods collapse.</p>" % (head, "".join(rows), orow, cap, R["N_train"])
(HERE / "owwin3_table_uncap.html").write_text(table("uncap"))
(HERE / "owwin3_table_cap.html").write_text(table("cap"))

def png(name, c):
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); MK = {"square": "s", "circle": "o"}
    for s in c["series"]:
        n = s["id"]; ls = "--" if n.endswith("-X-X") else (":" if n == "Oracle" else "-")
        ax.plot(c["x"], s["y"], color=s["color"], ls=ls, marker=MK.get(s.get("marker"), ""), ms=4, lw=s.get("width", 2.0), label=s["label"])
    for hl in c.get("hlines", []): ax.axhline(hl["y"], ls="--", lw=1, color=hl["color"])
    ax.set_xlabel(c["xlabel"]); ax.set_ylabel(c["ylabel"]); ax.set_ylim(c["ymin"], c["ymax"]); ax.legend(fontsize=6.5, ncol=2); ax.grid(alpha=.25); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / (name + ".png"), dpi=120); plt.close(fig)
for nm, c in W.items():
    if "series" in c: png(nm, c)

idx = ROOT / "index.html"; h = idx.read_text()
line = "var OWWIN3_CHARTS = %s;" % json.dumps(W, separators=(",", ":"))
if re.search(r'^var OWWIN3_CHARTS = .*;$', h, flags=re.M):
    h = re.sub(r'^var OWWIN3_CHARTS = .*;$', lambda m: line, h, count=1, flags=re.M)
else:
    h = re.sub(r'^(var OWWIN2_CHARTS = .*;)$', lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
calls = "\n".join("renderChart('ichart-ow3-%s',OWWIN3_CHARTS.%s);" % (k, k) for k in W)
h = re.sub(r"\nrenderChart\('ichart-ow3-[^\n]*", "", h)
if "renderChart('ichart-ow3-outcome'" not in h:
    anchor = "renderChart('ichart-ow2-outcome',OWWIN2_CHARTS.outcome);"
    assert anchor in h, "owwin anchor missing"; h = h.replace(anchor, anchor + "\n" + calls, 1)
for ph, fn, bid in [("OW3_TBL_UNCAP", "owwin3_table_uncap.html", "restab-ow3-uncap"), ("OW3_TBL_CAP", "owwin3_table_cap.html", "restab-ow3-cap")]:
    tbl = (HERE / fn).read_text().strip()
    if h.count(ph) == 1:
        h = h.replace(ph, tbl)
    else:  # already filled -> refresh the table box contents
        h = re.sub(r'(<div class="restab-box" id="%s">\s*).*?(\s*</div>)' % bid, lambda mm: mm.group(1) + tbl + mm.group(2), h, count=1, flags=re.S)
idx.write_text(h)
print("spliced OWWIN3 (%d charts) + tables" % len(W))
for reg in ("uncap", "cap"):
    M = R["regimes"][reg]["mean"]; print("[%s] DR: X-X=%.3f O-X=%s O-W=%s" % (reg, M["DoublyRobust-X-X"][0], M["DoublyRobust-O-X"], M["DoublyRobust-O-W"]))
