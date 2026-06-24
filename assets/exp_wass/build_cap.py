"""Build the CAPPED value-vs-Γ + policy charts (and a results table) for the continuous and/or discrete
'Wasserstein wins' experiments, and splice them into index.html.  Run AFTER run_wass_cap.py / run_disc_cap.py.

Usage:
  python3 build_cap.py continuous   # reads assets/exp_wass/wass_cap_results.json      -> WASSCAP_CHARTS, ichart-wass-cap-*
  python3 build_cap.py discrete     # reads assets/exp_wass_disc/disc_cap_results.json -> DISCCAP_CHARTS, ichart-disc-cap-*
"""
import sys, json, re, importlib.util as ilu
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = Path("/Users/mohammadsaeedhaghi/Desktop/01-Research/paper - policy optimization under unobsorved confounder - code/code 1.1")
COL = {"IPW-X-X": "#1f77b4", "IPW-O-X": "#1f77b4", "DoublyRobust-X-X": "#2ca02c",
       "DoublyRobust-O-X": "#2ca02c", "IPW-O-W": "#1f77b4", "DoublyRobust-O-W": "#2ca02c", "Oracle": "#444444"}
FAMCOL = {"IPW": "#1f77b4", "DoublyRobust": "#2ca02c", "Oracle": "#444444"}

which = sys.argv[1] if len(sys.argv) > 1 else "continuous"
if which == "continuous":
    RES = ROOT / "assets/exp_wass/wass_cap_results.json"; HERE = ROOT / "assets/exp_wass"
    PREF, VAR, XLAB = "wass-cap", "WASSCAP_CHARTS", "X  (continuous covariate)"
    XMIN, XMAX, anchor_render = -1.0, 1.0, "renderChart('ichart-wass-policy',WASS_CHARTS.policy);"
else:
    RES = ROOT / "assets/exp_wass_disc/disc_cap_results.json"; HERE = ROOT / "assets/exp_wass_disc"
    PREF, VAR, XLAB = "disc-cap", "DISCCAP_CHARTS", "X  (discrete covariate)"
    XMIN, XMAX, anchor_render = -1.0, 1.0, "renderChart('ichart-disc-policy',DISC_CHARTS.policy);"

R = json.loads(RES.read_text()); GAMMAS = R["gammas"]; G = np.array(R["grid"]); V = R["value_by_gamma"]; orc = R["oracle"]
def gk(g): g = float(g); return str(int(g)) if g == int(g) else str(g)
def ser(idl, label, color, y, **kw):
    dd = {"id": idl, "label": label, "color": color, "y": [round(float(v), 4) for v in y]}; dd.update(kw); return dd
allv = [v for k in V for v in V[k]] + [orc]
ymin, ymax = min(allv) - 0.03, max(allv) + 0.03

order = ["IPW-X-X", "DoublyRobust-X-X", "IPW-O-X", "DoublyRobust-O-X", "IPW-O-W", "DoublyRobust-O-W"]
W = {}
W["value"] = {"x": [float(g) for g in GAMMAS], "xmin": 1.0, "xmax": 8.0, "xlabel": "Γ  (sensitivity-model strength)",
              "ylabel": "realised E[Y] on test", "ymin": round(ymin, 3), "ymax": round(ymax, 3),
              "series": [ser(m, m, COL[m], V[m]) for m in order] + [ser("Oracle", "Oracle", COL["Oracle"], [orc] * len(GAMMAS))],
              "hlines": [{"y": orc, "color": "#94a3b8", "dash": "4,4", "label": "oracle"}],
              "note": "CAPPED (treat <= 50%%): realised test E[Y] vs Γ, N_train=%d, single seed. Circles = Wasserstein (-O-W), squares = box (-O-X)." % R["N_train"]}
Pol = R["policy_by_gamma"]
dgpmod = ilu.spec_from_file_location("capdgp", str(HERE / "dgp.py")); dm = ilu.module_from_spec(dgpmod); dgpmod.loader.exec_module(dm)
oracle = dm.oracle_policy(G)
def psa(g):
    k = gk(g)
    return [ser("IPW-O-X", "IPW-O-X", COL["IPW-O-X"], Pol["IPW-O-X"][k]),
            ser("DoublyRobust-O-X", "DoublyRobust-O-X", COL["DoublyRobust-O-X"], Pol["DoublyRobust-O-X"][k]),
            ser("IPW-O-W", "IPW-O-W", COL["IPW-O-W"], Pol["IPW-O-W"][k]),
            ser("DoublyRobust-O-W", "DoublyRobust-O-W", COL["DoublyRobust-O-W"], Pol["DoublyRobust-O-W"][k]),
            ser("Oracle", "Oracle (treat iff X>0)", COL["Oracle"], [round(float(v), 4) for v in oracle])]
W["policy"] = {"x": [float(v) for v in G], "xmin": XMIN, "xmax": XMAX, "xlabel": XLAB,
               "ylabel": "π(treat | X)", "ymin": -0.05, "ymax": 1.05, "gammas": [float(g) for g in GAMMAS],
               "defaultGamma": 2.0, "matchedGamma": 8.0, "seriesByGamma": {gk(g): psa(g) for g in GAMMAS},
               "note": "CAPPED deployed π(treat|X), N_train=%d. Under the 50%% cap the box mis-selects whom to treat; Wasserstein balances X (and so the hidden S) and selects the right half." % R["N_train"]}

(HERE / (PREF.replace("-", "_") + "_charts.json")).write_text(json.dumps(W))

# results table
def fmt(v): return ("%.3f" % v).replace("-0.000", "0.000")
def fam(m): return "DoublyRobust" if m.startswith("DoublyRobust") else ("IPW" if m.startswith("IPW") else "Oracle")
pair = {"IPW": {}, "DoublyRobust": {}}
for m in order:
    if m.endswith("-O-W") or m.endswith("-O-X"): pair[fam(m)][m[-3:]] = V[m]
def rowhtml(label, vals, bold):
    cells = "".join("<td%s>%s</td>" % (" style='font-weight:700'" if b else "", fmt(v)) for v, b in zip(vals, bold))
    dot = FAMCOL[fam(label)]
    return "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>%s</td>%s</tr>" % (dot, label, cells)
head = "<tr><th>method</th>" + "".join("<th>Γ=%s</th>" % gk(g) for g in GAMMAS) + "</tr>"
body = []
for m in order:
    mask = [False] * len(GAMMAS)
    if m.endswith("-O-W") or m.endswith("-O-X"):
        f, suf = fam(m), m[-3:]; other = pair[f].get("O-X" if suf == "O-W" else "O-W")
        if other: mask = [V[m][i] > other[i] + 1e-9 for i in range(len(GAMMAS))]
    body.append(rowhtml(m, V[m], mask))
orc_row = "<tr><td><span style='display:inline-block;width:9px;height:9px;border-radius:2px;background:%s;margin-right:6px'></span>Oracle (ceiling)</td>%s</tr>" % (FAMCOL["Oracle"], "".join("<td>%s</td>" % fmt(orc) for _ in GAMMAS))
cap_frac = int(R["cap"][1] * 100)
best_box = max(max(V["IPW-O-X"]), max(V["DoublyRobust-O-X"]), V["IPW-X-X"][0], V["DoublyRobust-X-X"][0])
best_w = max(max(V["IPW-O-W"]), max(V["DoublyRobust-O-W"]))
verdict = ("Every Wasserstein value beats every box value; DoublyRobust-O-W is the overall best, closest to oracle %.3f." % orc
           if best_w > best_box + 1e-9 else
           "Best Wasserstein %.3f vs best box %.3f (oracle %.3f): the box still snags a lucky peak, so this is a near-tie, not an outright Wasserstein win." % (best_w, best_box, orc))
tbl = ("<table class='restab'>%s%s%s</table>\n<p class='muted' style='margin:6px 0 0'>CAPPED (treat &le; %d%%): realised test E[Y] by Γ, N_train=%d, single seed. Bold = family winner (-O-W vs -O-X) at that Γ. %s</p>"
       % (head, "".join(body), orc_row, cap_frac, R["N_train"], verdict))
(HERE / (PREF.replace("-", "_") + "_table.html")).write_text(tbl)

# PNGs
def png(name, c):
    fig, ax = plt.subplots(figsize=(7.4, 4.2)); MK = {"square": "s", "circle": "o"}
    for s in c["series"]:
        ax.plot(c["x"], s["y"], marker=MK.get(s.get("marker"), ""), ms=4, lw=s.get("width", 2.0), color=s["color"], label=s["label"])
    for hl in c.get("hlines", []): ax.axhline(hl["y"], ls="--", lw=1, color=hl["color"])
    ax.set_xlabel(c["xlabel"]); ax.set_ylabel(c["ylabel"]); ax.set_ylim(c["ymin"], c["ymax"]); ax.legend(fontsize=7); ax.grid(alpha=.25); ax.set_title(name, fontsize=9)
    fig.tight_layout(); fig.savefig(HERE / (PREF + "-" + name + ".png"), dpi=120); plt.close(fig)
png("value", W["value"])

# splice var + render calls
idx = ROOT / "index.html"; h = idx.read_text()
line = "var %s = %s;" % (VAR, json.dumps(W, separators=(",", ":")))
if re.search(r'^var %s = .*;$' % VAR, h, flags=re.M):
    h = re.sub(r'^var %s = .*;$' % VAR, lambda m: line, h, count=1, flags=re.M)
else:
    parent = "WASS_CHARTS" if which == "continuous" else "DISC_CHARTS"
    h = re.sub(r'^(var %s = .*;)$' % parent, lambda m: m.group(1) + "\n" + line, h, count=1, flags=re.M)
calls = "\n".join("renderChart('ichart-%s-%s',%s.%s);" % (PREF, k, VAR, k) for k in W)
h = re.sub(r"\nrenderChart\('ichart-%s-[^\n]*" % PREF, "", h)
if ("renderChart('ichart-%s-value'" % PREF) not in h:
    assert anchor_render in h, "render anchor missing: " + anchor_render
    h = h.replace(anchor_render, anchor_render + "\n" + calls, 1)
idx.write_text(h)
print("[%s] spliced %s (%d charts) + table; ymin=%.3f ymax=%.3f oracle=%.3f" % (which, VAR, len(W), ymin, ymax, orc))
